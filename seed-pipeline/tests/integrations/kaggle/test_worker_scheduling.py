import asyncio
import math

import pytest

from seed_pipeline.integrations.kaggle.workers.scheduling import (
    stream_map_ordered,
)


async def _wait_for(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached")


@pytest.mark.asyncio
async def test_streaming_pool_has_no_cross_resource_barrier():
    release_slow = asyncio.Event()
    started: list[tuple[int, str]] = []

    async def operation(resource, _index, item):
        started.append((resource, item))
        if item == "slow":
            await release_slow.wait()
        return item.upper()

    task = asyncio.create_task(
        stream_map_ordered(
            ["slow", "fast-1", "fast-2"],
            [0, 1],
            1,
            operation,
            deadline=math.inf,
        )
    )
    await _wait_for(lambda: (1, "fast-2") in started)
    release_slow.set()

    assert (await task).results == ["SLOW", "FAST-1", "FAST-2"]


@pytest.mark.asyncio
async def test_streaming_pool_respects_per_resource_concurrency():
    active = {0: 0, 1: 0}
    peak = {0: 0, 1: 0}

    async def operation(resource, _index, item):
        active[resource] += 1
        peak[resource] = max(peak[resource], active[resource])
        await asyncio.sleep(0.001)
        active[resource] -= 1
        return item

    result = await stream_map_ordered(
        list(range(12)),
        [0, 1],
        2,
        operation,
        deadline=math.inf,
    )

    assert result.results == list(range(12))
    assert peak == {0: 2, 1: 2}


@pytest.mark.asyncio
async def test_streaming_pool_delivers_completed_callback_out_of_order():
    completed: list[list[tuple[int, str]]] = []

    async def operation(_resource, _index, item):
        if item == "first":
            await asyncio.sleep(0.01)
        return item

    result = await stream_map_ordered(
        ["first", "second"],
        [0, 1],
        1,
        operation,
        deadline=math.inf,
        on_completed=lambda batch: completed.append(batch),
    )

    assert result.results == ["first", "second"]
    assert completed
    assert completed[0][0][0] == 1


@pytest.mark.asyncio
async def test_streaming_pool_cancels_outstanding_work_after_error():
    cancelled = asyncio.Event()

    async def operation(_resource, _index, item):
        if item == "fail":
            raise ValueError("broken")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(ValueError, match="broken"):
        await stream_map_ordered(
            ["wait", "fail"], [0, 1], 1, operation, deadline=math.inf
        )

    assert cancelled.is_set()
