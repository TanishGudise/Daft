from __future__ import annotations

import asyncio
import math
import os
import random
import weakref
from typing import TYPE_CHECKING, Any, Concatenate, TypeVar

from daft import DataType
from daft.dependencies import np, pa
from daft.errors import RetryAfterError
from daft.expressions.expressions import Expression
from daft.series import Series
from daft.udf.metrics import _metrics_context

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from daft.daft import OperatorMetrics, PyDataType, PySeries

    from .udf_v2 import ClsBase

C = TypeVar("C")


def replace_expressions_with_evaluated_args(
    original_args: tuple[tuple[Any, ...], dict[str, Any]],
    evaluated_args: list[Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args, kwargs = original_args
    new_args = tuple(evaluated_args.pop(0) if isinstance(arg, Expression) else arg for arg in args)
    new_kwargs = {key: (evaluated_args.pop(0) if isinstance(arg, Expression) else arg) for key, arg in kwargs.items()}
    return new_args, new_kwargs


# Same default and environment variable as the in-flight task limit of the async UDF sink.
_DEFAULT_MAX_ASYNC_CALLS = 64
# Same backoff as `retry_with_backoff` in src/daft-dsl/src/python_udf/retry.rs, applied per row.
_INITIAL_RETRY_DELAY_MS = 100
_MAX_RETRY_DELAY_MS = 60_000

# Every async UDF call in a process runs on Daft's single background event loop, so these semaphores are only
# ever used from that one loop.
_call_limits: weakref.WeakKeyDictionary[Any, dict[tuple[Any, int], asyncio.Semaphore]] = weakref.WeakKeyDictionary()


def _call_limit(cls: ClsBase[C], method: Any, max_concurrency: int | None) -> asyncio.Semaphore:
    """Returns the semaphore that caps concurrent calls of `method` on this UDF object in this process.

    Every batch that calls the method on the same object shares it, including batches of later queries.
    """
    if max_concurrency is None:
        try:
            value = int(os.environ.get("DAFT_MAX_ASYNC_UDF_INFLIGHT_TASKS", ""))
        except ValueError:
            value = -1
        max_concurrency = max(value, 1) if value >= 0 else _DEFAULT_MAX_ASYNC_CALLS
    limits = _call_limits.setdefault(cls, {})
    key = (method, max_concurrency)
    if key not in limits:
        limits[key] = asyncio.Semaphore(max_concurrency)
    return limits[key]


def _retry_after_ms(error: Exception) -> int | None:
    if not isinstance(error, RetryAfterError):
        return None
    try:
        seconds = float(error.retry_after)
    except (TypeError, ValueError):
        return None
    return math.ceil(seconds * 1000) if math.isfinite(seconds) and seconds >= 0 else None


async def _call_with_retries(call: Callable[[], Coroutine[Any, Any, Any]], max_retries: int) -> Any:
    delay_ms = _INITIAL_RETRY_DELAY_MS
    for _ in range(max_retries):
        try:
            return await call()
        except Exception as e:
            retry_after_ms = _retry_after_ms(e)
            if retry_after_ms is not None:
                delay_ms = max(min(retry_after_ms, _MAX_RETRY_DELAY_MS), delay_ms)
        # ±25% jitter, as in the Rust helper.
        await asyncio.sleep((delay_ms * 3 // 4 + random.randint(0, delay_ms // 2)) / 1000)
        delay_ms = min(delay_ms * 2, _MAX_RETRY_DELAY_MS)

    try:
        return await call()
    except RetryAfterError as e:
        # When retries run out, raise the original exception rather than the RetryAfterError wrapper.
        if e.__cause__ is None:
            raise
        original = e.__cause__
    raise original


async def call_async_func_batched(
    cls: ClsBase[C],
    method: Callable[Concatenate[C, ...], Any],
    return_dtype: PyDataType,
    original_args: tuple[tuple[Any, ...], dict[str, Any]],
    evaluated_args_list: list[list[Any]],
    max_concurrency: int | None = None,
    max_retries: int = 0,
    raise_on_error: bool = True,
) -> tuple[PySeries, OperatorMetrics, list[Exception]]:
    """Calls an async row-wise UDF on every row of a batch and returns the outputs in the batch's row order.

    At most `max_concurrency` calls run at once, counted across every batch that calls this method on the same UDF
    object in the process. Each row is retried on its own. If `raise_on_error` is False, a row that still fails
    becomes None and its error is returned; otherwise the first such error is raised and this batch's other calls
    are cancelled.
    """
    bound_method = cls._daft_bind_method(method)
    limit = _call_limit(cls, method, max_concurrency)

    async def call_row(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        async def attempt() -> Any:
            async with limit:
                return await bound_method(*args, **kwargs)

        return await _call_with_retries(attempt, max_retries)

    rows = [
        replace_expressions_with_evaluated_args(original_args, evaluated_args) for evaluated_args in evaluated_args_list
    ]

    dtype = DataType._from_pydatatype(return_dtype)
    with _metrics_context() as metrics:
        tasks = [asyncio.create_task(call_row(args, kwargs)) for args, kwargs in rows]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=not raise_on_error)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    outputs: list[Any] = []
    errors: list[Exception] = []
    for result in results:
        if isinstance(result, Exception):
            outputs.append(None)
            errors.append(result)
        elif isinstance(result, BaseException):
            raise result
        else:
            outputs.append(result)

    series = Series.from_pylist(outputs, dtype=dtype)._series
    return series, metrics, errors


def call_func(
    cls: ClsBase[C],
    method: Callable[Concatenate[C, ...], Any],
    original_args: tuple[tuple[Any, ...], dict[str, Any]],
    evaluated_args: list[Any],
) -> tuple[Any, OperatorMetrics]:
    """Called from Rust to evaluate a Python scalar UDF. Returns a list of Python objects."""
    args, kwargs = replace_expressions_with_evaluated_args(original_args, evaluated_args)

    bound_method = cls._daft_bind_method(method)

    with _metrics_context() as metrics:
        output = bound_method(*args, **kwargs)

    return output, metrics


def call_batch_func(
    cls: ClsBase[C],
    method: Callable[Concatenate[C, ...], Series],
    original_args: tuple[tuple[Any, ...], dict[str, Any]],
    evaluated_args: list[PySeries],
) -> tuple[PySeries, OperatorMetrics]:
    """Called from Rust to evaluate a Python batch UDF. Returns a PySeries."""
    evaluated_args_series = [Series._from_pyseries(arg) for arg in evaluated_args]
    args, kwargs = replace_expressions_with_evaluated_args(original_args, evaluated_args_series)

    bound_method = cls._daft_bind_method(method)

    with _metrics_context() as metrics:
        output = bound_method(*args, **kwargs)

    if isinstance(output, Series):
        output_series = output
    elif isinstance(output, list):
        output_series = Series.from_pylist(output)
    elif np.module_available() and isinstance(output, np.ndarray):
        output_series = Series.from_numpy(output)
    elif pa.module_available() and isinstance(output, (pa.Array, pa.ChunkedArray)):
        output_series = Series.from_arrow(output)
    else:
        raise ValueError(f"Expected output to be a Series, list, numpy array, or pyarrow array, got {type(output)}")

    return output_series._series, metrics


async def call_batch_async(
    cls: ClsBase[C],
    method: Callable[Concatenate[C, ...], Coroutine[Any, Any, Series]],
    original_args: tuple[tuple[Any, ...], dict[str, Any]],
    evaluated_args: list[PySeries],
) -> tuple[PySeries, OperatorMetrics]:
    """Called from Rust to evaluate a Python batch UDF. Returns a PySeries."""
    evaluated_args_series = [Series._from_pyseries(arg) for arg in evaluated_args]
    args, kwargs = replace_expressions_with_evaluated_args(original_args, evaluated_args_series)

    bound_coroutine: Callable[..., Coroutine[Any, Any, Series]] = cls._daft_bind_coroutine_method(method)

    with _metrics_context() as metrics:
        output = await bound_coroutine(*args, **kwargs)

    if isinstance(output, Series):
        output_series = output
    elif isinstance(output, list):
        output_series = Series.from_pylist(output)
    elif np.module_available() and isinstance(output, np.ndarray):
        output_series = Series.from_numpy(output)
    elif pa.module_available() and isinstance(output, (pa.Array, pa.ChunkedArray)):
        output_series = Series.from_arrow(output)
    else:
        raise ValueError(f"Expected output to be a Series, list, numpy array, or pyarrow array, got {type(output)}")

    return output_series._series, metrics
