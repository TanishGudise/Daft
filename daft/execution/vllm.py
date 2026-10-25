from __future__ import annotations

import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray import ObjectRef

from daft.recordbatch import RecordBatch


class VLLMExecutor(ABC):
    @abstractmethod
    def submit(self, prefix: str, prompts: list[str], rows: RecordBatch) -> None:
        """Submit a batch of prompts and their corresponding rows to the executor, returning once all tasks are started."""

    @abstractmethod
    def poll(self) -> tuple[list[str], RecordBatch] | None:
        """Poll the executor for completed tasks. Returns a tuple of completed outputs and their corresponding rows, or None if no tasks are completed."""

    @abstractmethod
    def finished_submitting(self) -> None:
        """Call this when all tasks have been submitted."""

    @abstractmethod
    def all_tasks_finished(self) -> bool:
        """Check if all tasks have been completed and all results have been polled."""


class DummyVLLMExecutor(VLLMExecutor):
    def __init__(self, _model: str, _engine_args: dict[str, Any], _generate_args: dict[str, Any]):
        self.prompts: list[str] = []
        self.rows: list[RecordBatch] = []
        self._finished_submitting = False

    def submit(self, prefix: str, prompts: list[str], rows: RecordBatch) -> None:
        self.prompts.extend(prompts)
        self.rows.append(rows)

    def poll(self) -> tuple[list[str], RecordBatch] | None:
        if len(self.prompts) == 0:
            return None
        else:
            results = (self.prompts, RecordBatch.concat(self.rows))
            self.prompts = []
            self.rows = []
            return results

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        return self._finished_submitting and len(self.prompts) == 0


class BlockingVLLMExecutor(VLLMExecutor):
    def __init__(self, model: str, engine_args: dict[str, Any], generate_args: dict[str, Any]):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, **engine_args)

        self.sampling_params = generate_args.pop("sampling_params", SamplingParams())
        self.generate_args = generate_args

        self._finished_submitting = False

        self.completed_tasks: list[tuple[list[str], RecordBatch]] = []

    def submit(self, prefix: str, prompts: list[str], rows: RecordBatch) -> None:
        import threading

        results_container = {}

        def run_generate() -> None:
            results_container["results"] = self.llm.generate(prompts, self.sampling_params, **self.generate_args)

        generate_thread = threading.Thread(target=run_generate)
        generate_thread.start()
        generate_thread.join()
        results = results_container["results"]
        outputs = [r.outputs[0].text for r in results]

        self.completed_tasks.append((outputs, rows))

    def poll(self) -> tuple[list[str], RecordBatch] | None:
        if len(self.completed_tasks) == 0:
            return None

        completed_outputs = [output for outputs in (t[0] for t in self.completed_tasks) for output in outputs]
        completed_rows = [t[1] for t in self.completed_tasks]
        completed_rows_batch = RecordBatch.concat(completed_rows)

        self.completed_tasks = []
        return completed_outputs, completed_rows_batch

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        return self._finished_submitting and len(self.completed_tasks) == 0


class LocalVLLMExecutor(VLLMExecutor):
    def __init__(
        self, model: str, engine_args: dict[str, Any], generate_args: dict[str, Any], use_threading: bool = True
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        args = AsyncEngineArgs(model=model, **engine_args)

        self.llm = AsyncLLMEngine.from_engine_args(args)

        self.sampling_params = generate_args.pop("sampling_params", SamplingParams())
        self.generate_args = generate_args

        self.counter = 0
        self.counter_lock = threading.Lock()

        self.running_task_count = 0
        self.task_count_lock = threading.Lock()
        # First exception raised by a generate request; re-raised by poll() and all_tasks_finished().
        self.error: BaseException | None = None

        self.completed_tasks: deque[tuple[str, RecordBatch]] = deque()
        # On Ray, results and in-flight counts per RemoteVLLMExecutor (one per partition), so each partition
        # polls and waits for its own prompts only.
        self.completed_by_owner: defaultdict[str, deque[tuple[str, RecordBatch]]] = defaultdict(deque)
        self.running_by_owner: defaultdict[str, int] = defaultdict(int)
        # Holds references to submit_async's tasks until they finish.
        self.generate_tasks: set[asyncio.Task[None]] = set()

        self.last_log_time = time.perf_counter()
        self.last_log_lock = threading.Lock()
        self._periodic_log_started = False

        self.use_threading = use_threading
        if self.use_threading:
            # Create a dedicated event loop in a separate thread
            self.loop_ready = threading.Event()
            self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            self.loop_ready.wait()

        self._finished_submitting = False

    def _run_event_loop(self) -> None:
        """Run the event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()  # Signal that the loop is ready
        self.loop.run_forever()

    async def _periodic_log(self) -> None:
        while True:
            await asyncio.sleep(5)
            await self.llm.do_log_stats()
            with self.last_log_lock:
                self.last_log_time = time.perf_counter()
            with self.task_count_lock:
                if self.error is not None or (self._finished_submitting and self.running_task_count == 0):
                    return

    async def _generate(self, prompt: str, row: RecordBatch, owner: str | None = None) -> None:
        try:
            with self.counter_lock:
                id = self.counter
                self.counter += 1

            final_output = None
            async for output in self.llm.generate(prompt, self.sampling_params, str(id), **self.generate_args):
                final_output = output

            output_text: str = final_output.outputs[0].text  # type: ignore
            completed = self.completed_tasks if owner is None else self.completed_by_owner[owner]
            completed.append((output_text, row))
        except BaseException as e:
            # Nothing awaits this task, so keep the error for poll() to raise.
            with self.task_count_lock:
                if self.error is None:
                    self.error = e
            if not isinstance(e, Exception):
                raise
            return
        finally:
            with self.task_count_lock:
                self.running_task_count -= 1
                if owner is not None:
                    self.running_by_owner[owner] -= 1

        # Check if we need to log (in case periodic log isn't running in distributed mode)
        current_time = time.perf_counter()
        with self.last_log_lock:
            time_since_last_log = current_time - self.last_log_time
            if time_since_last_log >= 5.0:
                await self.llm.do_log_stats()
                self.last_log_time = time.perf_counter()

    def submit(self, _prefix: str | None, prompts: list[str], rows: RecordBatch) -> None:
        """Submit a batch of prompts and rows to the executor, returning once all tasks are started."""
        assert len(prompts) == len(rows), "Number of prompts and rows must match"

        if not self.use_threading:
            raise ValueError("Synchronous mode not supported when use_threading is False")

        with self.task_count_lock:
            self.running_task_count += len(prompts)

        if not self._periodic_log_started:
            self._periodic_log_started = True
            asyncio.run_coroutine_threadsafe(self._periodic_log(), self.loop)

        for i in range(len(prompts)):
            prompt = prompts[i]
            row = rows.slice(i, i + 1)
            asyncio.run_coroutine_threadsafe(self._generate(prompt, row), self.loop)

    async def submit_async(self, prompts: list[str], rows: RecordBatch, owner: str | None = None) -> None:
        assert len(prompts) == len(rows), "Number of prompts and rows must match"

        with self.task_count_lock:
            self.running_task_count += len(prompts)
            if owner is not None:
                self.running_by_owner[owner] += len(prompts)

        for i in range(len(prompts)):
            prompt = prompts[i]
            row = rows.slice(i, i + 1)
            task = asyncio.create_task(self._generate(prompt, row, owner))
            self.generate_tasks.add(task)
            task.add_done_callback(self.generate_tasks.discard)

    def poll(self, owner: str | None = None) -> tuple[list[str], RecordBatch] | None:
        """Poll the executor for completed tasks (on Ray, only those submitted by `owner`)."""
        with self.task_count_lock:
            error = self.error
        if error is not None:
            raise error

        # Drain the queues (no lock needed - queues are thread-safe)
        completed_outputs = []
        completed_rows = []
        completed = self.completed_tasks if owner is None else self.completed_by_owner.get(owner)
        if completed is None:
            return None

        while True:
            try:
                (output, row) = completed.popleft()
                completed_outputs.append(output)
                completed_rows.append(row)
            except IndexError:
                break

        if not completed_outputs:
            return None

        completed_rows_batch = RecordBatch.concat(completed_rows)
        return completed_outputs, completed_rows_batch

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        with self.task_count_lock:
            error = self.error
            finished = self._finished_submitting and self.running_task_count == 0 and len(self.completed_tasks) == 0
        if error is not None:
            raise error
        return finished

    def pending(self, owner: str) -> int:
        """Number of `owner`'s prompts that are still running or whose results haven't been polled."""
        with self.task_count_lock:
            error = self.error
            pending = self.running_by_owner.get(owner, 0) + len(self.completed_by_owner.get(owner, ()))
            if pending == 0:
                self.running_by_owner.pop(owner, None)
                self.completed_by_owner.pop(owner, None)
        if error is not None:
            raise error
        return pending


class RemoteVLLMExecutor(VLLMExecutor):
    """Submits one partition's prompts to the shared vLLM actors and collects only that partition's results."""

    def __init__(self, llm_actors: LLMActors):
        self.router_actor = llm_actors.router_actor
        self.llm_actors = llm_actors.llm_actors
        self.owner = uuid.uuid4().hex
        self.submitted: dict[int, int] = {}  # actor index -> rows sent to it
        self.received = 0
        self._finished_submitting = False

    @staticmethod
    def _get(refs: Any) -> Any:
        import ray

        try:
            return ray.get(refs)
        except (ray.exceptions.RayActorError, ray.exceptions.ActorUnschedulableError) as e:
            # Fail the query here. If a Ray actor error reaches the scheduler, it retries the task as if its
            # worker had failed, and the query can return with rows missing.
            raise RuntimeError(f"A vLLM actor failed or restarted during the query: {e}") from None

    def submit(self, prefix: str | None, prompts: list[str], rows: RecordBatch) -> None:
        route_to = self._get(self.router_actor.route.remote(prefix, len(prompts)))
        self._get(self.llm_actors[route_to].submit_async.remote(prompts, rows, self.owner))
        self.submitted[route_to] = self.submitted.get(route_to, 0) + len(prompts)

    def poll(self) -> tuple[list[str], RecordBatch] | None:
        actors = list(self.submitted)
        if not actors:
            return None
        results = self._get([self.llm_actors[actor].poll.remote(self.owner) for actor in actors])
        outputs: list[str] = []
        rows = []
        for result in results:
            if result is not None:
                outputs.extend(result[0])
                rows.append(result[1])
        if not outputs:
            return None
        self.received += len(outputs)
        return outputs, RecordBatch.concat(rows)

    def finished_submitting(self) -> None:
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        if not self._finished_submitting:
            return False
        actors = list(self.submitted)
        if any(self._get([self.llm_actors[actor].pending.remote(self.owner) for actor in actors])):
            return False
        sent = sum(self.submitted.values())
        if self.received < sent:
            raise RuntimeError(
                f"{sent - self.received} of {sent} prompts sent to vLLM actors never returned a result "
                "(a vLLM actor may have restarted)"
            )
        return True


class PrefixRouter:
    def __init__(
        self, llm_actors: list[ObjectRef[LocalVLLMExecutor]], load_balance_threshold: int, max_recent_prefixes: int = 8
    ):
        self.llm_actors = llm_actors
        self.loads = [0] * len(self.llm_actors)
        self.recent_prefixes: list[list[str | None]] = [[] for _ in range(len(self.llm_actors))]
        self.load_balance_threshold = load_balance_threshold
        self.max_recent_prefixes = max_recent_prefixes

    def route(self, prefix: str | None, batch_size: int) -> int:
        min_load = min(self.loads)

        if prefix is None:
            best_actor = self.loads.index(min_load)
        else:
            best_actor = None
            best_actor_recency = None
            for i in range(len(self.loads)):
                if self.loads[i] < min_load + self.load_balance_threshold:
                    try:
                        recency = self.recent_prefixes[i].index(prefix)
                    except ValueError:
                        recency = None

                    if best_actor_recency is None or recency is not None and recency < best_actor_recency:
                        best_actor = i
                        best_actor_recency = recency

        assert best_actor is not None, "No actor found for prefix"

        self.recent_prefixes[best_actor].insert(0, prefix)
        if len(self.recent_prefixes[best_actor]) > self.max_recent_prefixes:
            self.recent_prefixes[best_actor].pop()

        self.loads[best_actor] += batch_size

        return best_actor


class LLMActors:
    def __init__(
        self,
        model: str,
        engine_args: dict[str, Any],
        generate_args: dict[str, Any],
        gpus_per_actor: int,
        concurrency: int,
        load_balance_threshold: int,
    ):
        import ray

        LocalVLLMExecutorActor = ray.remote(num_gpus=gpus_per_actor, max_restarts=4)(LocalVLLMExecutor)
        PrefixRouterActor = ray.remote(PrefixRouter)

        self.llm_actors = [LocalVLLMExecutorActor.remote(model, engine_args, generate_args) for _ in range(concurrency)]
        self.router_actor = PrefixRouterActor.remote(self.llm_actors, load_balance_threshold)
