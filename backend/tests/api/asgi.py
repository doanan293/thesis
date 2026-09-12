"""Serve a FastAPI app in-process over ASGI for tests, without `starlette.testclient`.

The app's lifespan runs in the test's own event loop, so tests can await the same
services the app uses, and no thread portal is involved.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI


@asynccontextmanager
async def running(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield client
