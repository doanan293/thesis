from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ClientT = TypeVar("ClientT")


class ScheduledResultError(RuntimeError):
    """Raised when a partition does not return exactly its assigned indices."""


@dataclass(frozen=True)
class ScheduledBatch[OutputT]:
    results: list[OutputT]
    stopped_early: bool


async def stream_map_ordered[InputT, OutputT, ResourceT](
    items: Sequence[InputT],
    resources: Sequence[ResourceT],
    concurrency_per_resource: int,
    operation: Callable[[ResourceT, int, InputT], Awaitable[OutputT]],
    deadline: float,
    on_completed: Callable[[list[tuple[int, OutputT]]], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ScheduledBatch[OutputT]:
    """Stream a shared FIFO workload through long-lived resource workers."""
    if not resources:
        raise ValueError("at least one scheduler resource is required")
    if concurrency_per_resource < 1:
        raise ValueError("concurrency_per_resource must be positive")

    queue: asyncio.Queue[tuple[int, InputT]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))

    results: dict[int, OutputT] = {}
    stopped_early = False

    async def worker(resource: ResourceT) -> None:
        nonlocal stopped_early
        while True:
            if clock() >= deadline:
                stopped_early = stopped_early or not queue.empty()
                return
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                value = await operation(resource, index, item)
            except asyncio.CancelledError:
                queue.task_done()
                raise
            except Exception:
                queue.task_done()
                raise
            results[index] = value
            if on_completed is not None:
                on_completed([(index, value)])
            queue.task_done()

    tasks = [
        asyncio.create_task(worker(resource))
        for resource in resources
        for _ in range(concurrency_per_resource)
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    if not queue.empty() and clock() >= deadline:
        stopped_early = True
    return ScheduledBatch(
        [results[index] for index in sorted(results)],
        stopped_early,
    )


def _validate_partition[InputT, OutputT](
    partition: list[tuple[int, InputT]],
    returned: list[tuple[int, OutputT]],
    expected_indices: set[int],
) -> dict[int, OutputT]:
    by_index: dict[int, OutputT] = {}
    for index, value in returned:
        if index in by_index:
            raise ScheduledResultError("partition results are missing or duplicate")
        by_index[index] = value
    partition_indices = {index for index, _value in partition}
    if set(by_index) != expected_indices or partition_indices != expected_indices:
        raise ScheduledResultError("partition results are missing or duplicate")
    return by_index


def map_batches_ordered[InputT, OutputT, ClientT](
    items: Sequence[InputT],
    clients: Sequence[ClientT],
    per_client_batch_size: int,
    operation: Callable[[ClientT, list[tuple[int, InputT]]], list[tuple[int, OutputT]]],
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> ScheduledBatch[OutputT]:
    if not clients:
        raise ValueError("at least one scheduler client is required")
    if per_client_batch_size < 1:
        raise ValueError("per_client_batch_size must be positive")

    results: list[OutputT] = []
    outer_size = per_client_batch_size * len(clients)
    stopped_early = False
    for start in range(0, len(items), outer_size):
        if clock() >= deadline:
            stopped_early = True
            break
        outer_batch = [
            (index, items[index])
            for index in range(start, min(start + outer_size, len(items)))
        ]
        partitions: list[list[tuple[int, InputT]]] = [[] for _client in clients]
        for offset, item in enumerate(outer_batch):
            partitions[offset % len(clients)].append(item)
        active = [
            (client, partition)
            for client, partition in zip(clients, partitions, strict=True)
            if partition
        ]
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = [
                executor.submit(operation, client, partition)
                for client, partition in active
            ]
            partition_results = [future.result() for future in futures]

        by_index: dict[int, OutputT] = {}
        for (_client, partition), returned in zip(
            active, partition_results, strict=True
        ):
            partition_values = _validate_partition(
                partition,
                returned,
                {index for index, _value in partition},
            )
            overlap = set(by_index).intersection(partition_values)
            if overlap:
                raise ScheduledResultError(
                    "outer batch results contain duplicate indices"
                )
            by_index.update(partition_values)
        expected = {index for index, _value in outer_batch}
        if set(by_index) != expected:
            raise ScheduledResultError("outer batch results are missing or duplicate")
        results.extend(by_index[index] for index, _value in outer_batch)

    return ScheduledBatch(results, stopped_early)
