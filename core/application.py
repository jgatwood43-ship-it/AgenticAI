"""
core/application.py
───────────────────
Synchronous and asynchronous application service for terminal and Streamlit
frontends.

Design
------
A dedicated background asyncio event loop is created once. The reusable
MultiAgentWorkflow is constructed and used only on that loop.

This prevents:

    RuntimeError: Event loop is closed

which can occur when Streamlit repeatedly calls asyncio.run() while reusing
async resources created by an earlier loop.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, TypeVar

from core.workflow import MultiAgentWorkflow

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_workflow: MultiAgentWorkflow | None = None

_start_lock = threading.Lock()
_workflow_lock: asyncio.Lock | None = None


def _event_loop_worker(
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run the dedicated application event loop in its background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

    # Best-effort cleanup after loop.stop().
    pending = asyncio.all_tasks(loop)

    for task in pending:
        task.cancel()

    if pending:
        loop.run_until_complete(
            asyncio.gather(
                *pending,
                return_exceptions=True,
            )
        )

    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


def _ensure_application_loop() -> asyncio.AbstractEventLoop:
    """Start the persistent application event loop once."""
    global _loop
    global _loop_thread

    with _start_lock:
        if (
            _loop is not None
            and not _loop.is_closed()
            and _loop_thread is not None
            and _loop_thread.is_alive()
        ):
            return _loop

        loop = asyncio.new_event_loop()

        thread = threading.Thread(
            target=_event_loop_worker,
            args=(loop,),
            name="AgenticAIApplicationLoop",
            daemon=True,
        )

        thread.start()

        _loop = loop
        _loop_thread = thread

        print("[Application] Persistent event loop started.")

        return loop


async def _get_workflow_on_application_loop() -> MultiAgentWorkflow:
    """
    Create the workflow once on the same event loop used for all queries.
    """
    global _workflow
    global _workflow_lock

    if _workflow_lock is None:
        _workflow_lock = asyncio.Lock()

    async with _workflow_lock:
        if _workflow is None:
            print("[Application] Initializing persistent workflow...")

            # MultiAgentWorkflow construction is synchronous but may be
            # expensive. It must still occur on this persistent loop's thread
            # because some contained libraries bind async resources to the
            # currently active thread/loop.
            _workflow = MultiAgentWorkflow()

            print("[Application] Persistent workflow initialized.")

    return _workflow


def _submit_coroutine(
    coroutine: Coroutine[Any, Any, T],
) -> T:
    """Submit a coroutine to the persistent application loop and wait."""
    loop = _ensure_application_loop()

    future: Future[T] = asyncio.run_coroutine_threadsafe(
        coroutine,
        loop,
    )

    return future.result()


async def run_agent_query_async(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Execute one workflow query from asynchronous code.

    When called from outside the application loop, the work is safely submitted
    to that loop. When already running on the application loop, execution is
    performed directly.
    """
    cleaned_query = str(query or "").strip()

    if not cleaned_query:
        raise ValueError("The query cannot be empty.")

    application_loop = _ensure_application_loop()

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is application_loop:
        workflow = await _get_workflow_on_application_loop()

        state = await workflow.run(
            cleaned_query,
            conversation_history=conversation_history,
        )
    else:
        wrapped_future = asyncio.run_coroutine_threadsafe(
            _run_workflow_query(
                cleaned_query,
                conversation_history,
            ),
            application_loop,
        )

        state = await asyncio.wrap_future(wrapped_future)

    if state.error:
        raise RuntimeError(state.error)

    if not state.answer:
        return "The workflow completed but did not produce an answer."

    return state.answer


async def _run_workflow_query(
    query: str,
    conversation_history: list[dict[str, str]] | None,
):
    """Run one query on the persistent application loop."""
    workflow = await _get_workflow_on_application_loop()

    print(f"[Application] Submitting query: {query}")

    return await workflow.run(
        query,
        conversation_history=conversation_history,
    )


def run_agent_query(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Execute one workflow query from synchronous code such as Streamlit.

    Do not replace this with asyncio.run().
    """
    cleaned_query = str(query or "").strip()

    if not cleaned_query:
        raise ValueError("The query cannot be empty.")

    state = _submit_coroutine(
        _run_workflow_query(
            cleaned_query,
            conversation_history,
        )
    )

    if state.error:
        raise RuntimeError(state.error)

    if not state.answer:
        return "The workflow completed but did not produce an answer."

    return state.answer


def shutdown_application() -> None:
    """Stop the persistent application loop during interpreter shutdown."""
    global _loop
    global _loop_thread
    global _workflow
    global _workflow_lock

    loop = _loop
    thread = _loop_thread

    if loop is None or loop.is_closed():
        return

    loop.call_soon_threadsafe(loop.stop)

    if (
        thread is not None
        and thread.is_alive()
        and threading.current_thread() is not thread
    ):
        thread.join(timeout=2.0)

    _loop = None
    _loop_thread = None
    _workflow = None
    _workflow_lock = None


atexit.register(shutdown_application)
