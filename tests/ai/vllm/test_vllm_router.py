from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

ray = pytest.importorskip("ray")

from daft.execution.vllm import LocalVLLMExecutor, PrefixRouter, RemoteVLLMExecutor
from daft.recordbatch import RecordBatch


class FakeAsyncLLMEngine:
    """Stands in for `vllm.AsyncLLMEngine`: echoes prompts in upper case."""

    @classmethod
    def from_engine_args(cls, args):
        return cls()

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        yield types.SimpleNamespace(outputs=[types.SimpleNamespace(text=prompt.upper())])

    async def do_log_stats(self):
        pass


class FakeActorHandle:
    """Calls the wrapped object directly where a Ray actor handle would call `.remote()`.

    Coroutine methods (like `LocalVLLMExecutor.submit_async`) run on the object's own event loop, as they would on a
    Ray async actor. Replacing `obj` simulates Ray restarting the actor with fresh state; setting `failure` makes
    every call return that error, which the patched `ray.get` raises.
    """

    def __init__(self, obj):
        self.obj = obj
        self.failure = None

    def __getattr__(self, name):
        obj = self.obj
        method = getattr(obj, name)

        def remote(*args, **kwargs):
            if self.failure is not None:
                return self.failure
            result = method(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run_coroutine_threadsafe(result, obj.loop).result(timeout=10)
            return result

        return types.SimpleNamespace(remote=remote)


@pytest.fixture
def fake_vllm(monkeypatch):
    module = types.ModuleType("vllm")
    module.AsyncEngineArgs = lambda model, **kwargs: types.SimpleNamespace(model=model, **kwargs)
    module.AsyncLLMEngine = FakeAsyncLLMEngine
    module.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)

    def get(value):
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, BaseException):
                raise item
        return value

    monkeypatch.setattr(ray, "get", get)


@pytest.fixture
def llm_actors(fake_vllm):
    executors = []

    def new_executor():
        executor = LocalVLLMExecutor("fake-model", {}, {})
        executors.append(executor)
        return executor

    actors = [FakeActorHandle(new_executor()) for _ in range(2)]
    router = FakeActorHandle(PrefixRouter(actors, load_balance_threshold=256))
    yield types.SimpleNamespace(router_actor=router, llm_actors=actors, new_executor=new_executor)
    for executor in executors:
        executor.loop.call_soon_threadsafe(executor.loop.stop)


def drain(executor):
    """Poll a partition's executor until it reports that it is finished, like `VLLMSink::finalize`."""
    outputs = []
    for _ in range(400):
        polled = executor.poll()
        if polled is not None:
            outputs.extend(polled[0])
        if executor.all_tasks_finished():
            return outputs
        time.sleep(0.005)
    raise AssertionError("partition never finished")


def submit(executor, prompts):
    executor.submit(None, prompts, RecordBatch.from_pydict({"p": prompts}))
    executor.finished_submitting()


def partition_prompts(num_partitions, rows_per_partition):
    return [[f"p{part} row {i}" for i in range(rows_per_partition)] for part in range(num_partitions)]


def expected(partitions):
    return sorted(p.upper() for prompts in partitions for p in prompts)


def test_partitions_run_one_after_another_return_every_row(llm_actors):
    partitions = partition_prompts(4, 10)
    outputs = []
    for prompts in partitions:
        executor = RemoteVLLMExecutor(llm_actors)
        submit(executor, prompts)
        outputs.extend(drain(executor))

    assert sorted(outputs) == expected(partitions)


def test_partition_finishes_without_waiting_for_a_later_partition(llm_actors):
    # Partitions share one sink node, so a later partition only submits after the earlier one has finished.
    first, second = partition_prompts(2, 10)
    first_executor = RemoteVLLMExecutor(llm_actors)
    second_executor = RemoteVLLMExecutor(llm_actors)

    submit(first_executor, first)
    outputs = drain(first_executor)
    submit(second_executor, second)
    outputs += drain(second_executor)

    assert sorted(outputs) == expected([first, second])


def test_overlapping_partitions_return_every_row_once(llm_actors):
    partitions = partition_prompts(4, 10)
    executors = [RemoteVLLMExecutor(llm_actors) for _ in partitions]
    for executor, prompts in zip(executors, partitions):
        submit(executor, prompts)

    outputs = [output for executor in executors for output in drain(executor)]
    assert sorted(outputs) == expected(partitions)


def test_actor_error_fails_the_query_with_a_plain_error(llm_actors):
    executor = RemoteVLLMExecutor(llm_actors)
    submit(executor, partition_prompts(1, 10)[0])

    # The first call after an actor dies fails with a Ray actor error. It must not escape as one: the scheduler
    # would retry the task as if the worker had died.
    for actor in llm_actors.llm_actors:
        actor.failure = ray.exceptions.ActorUnavailableError("actor is restarting", None)

    with pytest.raises(RuntimeError, match="vLLM actor failed or restarted") as excinfo:
        drain(executor)
    assert not isinstance(excinfo.value, ray.exceptions.RayError)


def test_actor_restarted_with_fresh_state_raises_instead_of_dropping_rows(llm_actors):
    executor = RemoteVLLMExecutor(llm_actors)
    submit(executor, partition_prompts(1, 10)[0])

    # A restarted actor that no longer has this partition's prompts.
    for actor in llm_actors.llm_actors:
        actor.obj = llm_actors.new_executor()

    with pytest.raises(RuntimeError, match="never returned a result"):
        drain(executor)
