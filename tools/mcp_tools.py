"""
tools/mcp_tools.py
──────────────────
Thread-safe LlamaIndex FunctionTool wrappers for persistent Database-MCP
servers used by both terminal and Streamlit execution.

Design goals
------------
* One MCP subprocess per database type for the life of the Python process.
* Safe reuse across Streamlit reruns and across multiple agents.
* Serialized stdin/stdout access so concurrent agents cannot interleave JSON-RPC.
* Unique JSON-RPC request IDs.
* Restart only after transport/process failure—not after a normal MCP/SQL error.
* Always terminate an invalidated process before removing its session.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import re
import selectors
import subprocess
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llama_index.core.tools import FunctionTool

from config.settings import settings as db_config

MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_STARTUP_TIMEOUT_SECONDS = 30.0
MCP_RESPONSE_TIMEOUT_SECONDS = 240.0
MCP_SHUTDOWN_TIMEOUT_SECONDS = 5.0

_READ_ONLY_SQL = re.compile(
    r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN|WITH)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class MCPSession:
    """One persistent MCP subprocess and its synchronized I/O channel."""

    db_type: str
    process: subprocess.Popen[str]
    io_lock: threading.Lock = field(default_factory=threading.Lock)

    def is_alive(self) -> bool:
        return self.process.poll() is None


class MCPServerError(RuntimeError):
    """
    An error returned by the MCP server.

    This is an application/database error, not necessarily a broken transport.
    The persistent MCP process should normally remain available afterward.
    """


class MCPTransportError(RuntimeError):
    """The MCP subprocess or JSON-RPC transport is unusable."""


class MCPConnectionManager:
    """
    Process-wide manager for persistent MySQL and PostgreSQL MCP subprocesses.

    A single module-level instance is shared by every imported FunctionTool,
    every workflow agent, Streamlit reruns, and terminal callers in this Python
    process.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, MCPSession] = {}
        self._sessions_lock = threading.RLock()
        self._request_ids = itertools.count(1)
        atexit.register(self.close_all_sessions)

    def next_request_id(self) -> int:
        """Return a process-wide unique JSON-RPC request ID."""
        with self._sessions_lock:
            return next(self._request_ids)

    def get_session(self, db_type: str) -> MCPSession:
        """
        Return the live session for *db_type*, starting it when necessary.
        """
        normalized = self._normalize_db_type(db_type)

        with self._sessions_lock:
            existing = self._sessions.get(normalized)

            if existing is not None and existing.is_alive():
                return existing

            if existing is not None:
                self._terminate_session(existing)
                self._sessions.pop(normalized, None)

            session = self._start_session(normalized)
            self._sessions[normalized] = session
            return session

    def invalidate_session(
        self,
        db_type: str,
        expected_session: Optional[MCPSession] = None,
    ) -> None:
        """
        Terminate and remove a broken session.

        expected_session prevents one thread from destroying a newer replacement
        session created by another thread.
        """
        normalized = self._normalize_db_type(db_type)

        with self._sessions_lock:
            current = self._sessions.get(normalized)

            if current is None:
                return

            if expected_session is not None and current is not expected_session:
                return

            self._terminate_session(current)
            self._sessions.pop(normalized, None)

    def close_all_sessions(self) -> None:
        """Terminate all MCP subprocesses owned by this Python process."""
        with self._sessions_lock:
            for session in list(self._sessions.values()):
                self._terminate_session(session)

            self._sessions.clear()

    @staticmethod
    def _normalize_db_type(db_type: str) -> str:
        normalized = str(db_type).strip().lower()

        if normalized not in {"mysql", "postgres"}:
            raise ValueError(f"Unsupported database type: {db_type}")

        return normalized

    def _start_session(self, db_type: str) -> MCPSession:
        args = self._build_command(db_type)

        print(f"🚀 Launching persistent background MCP server for {db_type}...")

        # DEVNULL prevents an unread stderr pipe from filling and deadlocking the
        # subprocess. Set MCP_STDERR_LOG to a path to retain server diagnostics.
        stderr_target: Any = subprocess.DEVNULL
        stderr_log_path = os.getenv("MCP_STDERR_LOG", "").strip()
        stderr_file = None

        if stderr_log_path:
            stderr_file = open(stderr_log_path, "a", encoding="utf-8")
            stderr_target = stderr_file

        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_target,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        finally:
            # Popen duplicates the file descriptor, so the parent can close its
            # copy immediately.
            if stderr_file is not None:
                stderr_file.close()

        session = MCPSession(db_type=db_type, process=process)

        try:
            with session.io_lock:
                request_id = self.next_request_id()

                self._write_message(
                    session,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": MCP_PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {
                                "name": f"LlamaIndex-{db_type}-Agent",
                                "version": "1.0.0",
                            },
                        },
                    },
                )

                response = self._read_response(
                    session=session,
                    expected_id=request_id,
                    timeout=MCP_STARTUP_TIMEOUT_SECONDS,
                )

                if "error" in response:
                    raise MCPServerError(f"MCP initialize failed: {response['error']}")

                self._write_message(
                    session,
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                    },
                )

        except Exception:
            self._terminate_session(session)
            raise

        print(f"✅ {db_type.upper()} MCP server initialized and ready for reuse.")
        return session

    @staticmethod
    def _build_command(db_type: str) -> list[str]:
        if db_type == "postgres":
            return [
                "dbmcp",
                "stdio",
                "--db-backend",
                "postgres",
                "--db-host",
                str(db_config.pg_host),
                "--db-port",
                str(db_config.pg_port),
                "--db-name",
                str(db_config.pg_name),
                "--db-user",
                str(db_config.pg_user),
                "--db-password",
                str(db_config.pg_password),
            ]

        return [
            "dbmcp",
            "stdio",
            "--db-backend",
            "mysql",
            "--db-host",
            str(db_config.mysql_host),
            "--db-port",
            str(db_config.mysql_port),
            "--db-name",
            str(db_config.mysql_name),
            "--db-user",
            str(db_config.mysql_user),
            "--db-password",
            str(db_config.mysql_password),
        ]

    @staticmethod
    def _write_message(session: MCPSession, payload: Dict[str, Any]) -> None:
        process = session.process

        if not session.is_alive():
            raise MCPTransportError(f"The {session.db_type} MCP server is not running.")

        if process.stdin is None:
            raise MCPTransportError(
                f"The {session.db_type} MCP stdin stream is unavailable."
            )

        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise MCPTransportError(
                f"Could not write to the {session.db_type} MCP server: {exc}"
            ) from exc

    @staticmethod
    def _readline_with_timeout(
        process: subprocess.Popen[str],
        timeout: float,
        db_type: str,
    ) -> str:
        if process.stdout is None:
            raise MCPTransportError(f"The {db_type} MCP stdout stream is unavailable.")

        selector = selectors.DefaultSelector()

        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            ready = selector.select(timeout)

            if not ready:
                raise MCPTransportError(
                    f"Timed out after {timeout:.0f} seconds waiting for the "
                    f"{db_type} MCP server."
                )

            line = process.stdout.readline()

        finally:
            selector.close()

        if not line:
            return_code = process.poll()
            raise MCPTransportError(
                f"The {db_type} MCP server closed its output stream "
                f"(return code: {return_code})."
            )

        return line

    def _read_response(
        self,
        session: MCPSession,
        expected_id: int,
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Read messages until the response matching expected_id is received.

        Server notifications and unrelated messages are ignored. Because all
        requests for a session are serialized by io_lock, a different response
        ID indicates a protocol/configuration defect rather than concurrency.
        """
        while True:
            response_line = self._readline_with_timeout(
                process=session.process,
                timeout=timeout,
                db_type=session.db_type,
            )

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise MCPTransportError(
                    f"The {session.db_type} MCP server returned invalid JSON: "
                    f"{response_line[:300]!r}"
                ) from exc

            if not isinstance(response, dict):
                continue

            # Ignore JSON-RPC notifications or progress messages.
            if "id" not in response:
                continue

            if response.get("id") != expected_id:
                raise MCPTransportError(
                    f"Unexpected JSON-RPC response ID from {session.db_type}: "
                    f"expected {expected_id}, received {response.get('id')}."
                )

            return response

    @staticmethod
    def _terminate_session(session: MCPSession) -> None:
        process = session.process

        if process.poll() is not None:
            return

        print(f"🧹 Closing persistent {session.db_type} MCP background process...")

        process.terminate()

        try:
            process.wait(timeout=MCP_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=MCP_SHUTDOWN_TIMEOUT_SECONDS)


mcp_manager = MCPConnectionManager()


def _call_persistent_mcp(
    db_type: str,
    method: str,
    params: Dict[str, Any],
    *,
    allow_pool_restart: bool = True,
) -> Any:
    """
    Send one synchronized JSON-RPC request to a persistent MCP server.

    Behavior:
    - Transport failures terminate and remove the MCP subprocess.
    - SQL/schema errors keep the subprocess alive.
    - A database-pool timeout restarts the subprocess once and retries.
    """
    session = mcp_manager.get_session(db_type)
    request_id = mcp_manager.next_request_id()

    try:
        with session.io_lock:
            MCPConnectionManager._write_message(
                session,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
            )

            response = mcp_manager._read_response(
                session=session,
                expected_id=request_id,
                timeout=MCP_RESPONSE_TIMEOUT_SECONDS,
            )

    except MCPTransportError:
        mcp_manager.invalidate_session(
            db_type=db_type,
            expected_session=session,
        )
        raise

    if "error" not in response:
        return response.get("result")

    error = response["error"]
    error_text = str(error)
    error_text_lower = error_text.lower()

    # A pool timeout usually means the dbmcp process's internal connection
    # pool is no longer usable. Fully terminate that process, create a new
    # one, and retry this request once.
    if allow_pool_restart and "pool timed out" in error_text_lower:
        print(
            f"⚠️ {db_type.upper()} MCP connection pool timed out. "
            "Restarting the MCP subprocess once..."
        )

        mcp_manager.invalidate_session(
            db_type=db_type,
            expected_session=session,
        )

        return _call_persistent_mcp(
            db_type=db_type,
            method=method,
            params=params,
            allow_pool_restart=False,
        )

    # Ordinary SQL syntax, missing-table, missing-column, and authorization
    # errors do not require restarting the MCP subprocess.
    raise MCPServerError(f"MCP server returned error: {error}")


def _extract_tool_content(result: Any) -> str:
    """Normalize a dbmcp tools/call result into agent-readable text."""
    if isinstance(result, dict):
        content = result.get("content", [])

        if result.get("isError") is True:
            message = _content_to_text(content)
            return f"Database Error: {message or 'Unknown database error'}"

        if not content:
            return "Query executed successfully. No rows were returned."

        return json.dumps(content, indent=2, default=str)

    if result in (None, "", [], "[]"):
        return "Query executed successfully. No rows were returned."

    return json.dumps(result, indent=2, default=str)


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        messages = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                messages.append(str(item["text"]))
            else:
                messages.append(str(item))

        return "\n".join(messages)

    return str(content)


def _validate_read_only_sql(sql: str) -> str:
    cleaned = str(sql).strip()

    if not cleaned:
        raise ValueError("SQL cannot be empty.")

    if not _READ_ONLY_SQL.match(cleaned):
        raise ValueError(
            "Only SELECT, SHOW, DESCRIBE, DESC, EXPLAIN, and read-only WITH "
            "statements are allowed."
        )

    # Reject multiple statements while allowing a single trailing semicolon.
    without_trailing_semicolon = cleaned[:-1] if cleaned.endswith(";") else cleaned

    if ";" in without_trailing_semicolon:
        raise ValueError("Multiple SQL statements are not allowed.")

    return cleaned


def mysql_query_tool(sql: str) -> str:
    """Execute one completed, read-only SQL statement against MySQL."""
    try:
        safe_sql = _validate_read_only_sql(sql)
        result = _call_persistent_mcp(
            db_type="mysql",
            method="tools/call",
            params={
                "name": "readQuery",
                "arguments": {"query": safe_sql},
            },
        )
        return _extract_tool_content(result)

    except Exception as exc:
        _log_tool_exception("MYSQL", exc)
        return f"Database Error: {exc}"


def postgres_query_tool(sql: str) -> str:
    """Execute one completed, read-only SQL statement against PostgreSQL."""
    try:
        safe_sql = _validate_read_only_sql(sql)
        result = _call_persistent_mcp(
            db_type="postgres",
            method="tools/call",
            params={
                "name": "readQuery",
                "arguments": {"query": safe_sql},
            },
        )
        return _extract_tool_content(result)

    except Exception as exc:
        _log_tool_exception("POSTGRES", exc)
        return f"Database Error: {exc}"


def list_tables_tool(db_type: str) -> str:
    """List tables using dbmcp's native listTables tool."""
    try:
        result = _call_persistent_mcp(
            db_type=db_type,
            method="tools/call",
            params={"name": "listTables", "arguments": {}},
        )
        return _extract_tool_content(result)

    except Exception as exc:
        _log_tool_exception(db_type.upper(), exc)
        return f"Database Error: Could not list {db_type} tables: {exc}"


def describe_mysql_table_tool(table_name: str) -> str:
    """Safely describe one MySQL table."""
    safe_table = str(table_name).strip()

    if not safe_table or not safe_table.replace("_", "").isalnum():
        return f"Database Error: Invalid table name: {safe_table!r}"

    return mysql_query_tool(f"DESCRIBE `{safe_table}`")


def mysql_relationships_tool() -> str:
    """Return declared foreign-key relationships for the active MySQL schema."""
    sql = """
        SELECT
            kcu.TABLE_NAME AS child_table,
            kcu.COLUMN_NAME AS child_column,
            kcu.REFERENCED_TABLE_NAME AS parent_table,
            kcu.REFERENCED_COLUMN_NAME AS parent_column,
            kcu.CONSTRAINT_NAME AS constraint_name
        FROM information_schema.KEY_COLUMN_USAGE AS kcu
        WHERE kcu.CONSTRAINT_SCHEMA = DATABASE()
          AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
        ORDER BY
            kcu.TABLE_NAME,
            kcu.COLUMN_NAME
    """
    return mysql_query_tool(sql)


def _log_tool_exception(db_label: str, exc: Exception) -> None:
    print("\n" + "=" * 60)
    print(f"MCP TOOL ERROR — {db_label}")
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    print("=" * 60 + "\n", flush=True)


MYSQL_TOOL = FunctionTool.from_defaults(
    fn=mysql_query_tool,
    name="mysql_execute_read_query",
    description=(
        "Execute exactly one complete read-only MySQL statement. The `sql` "
        "argument must contain the full query. Allowed statements: SELECT, "
        "SHOW, DESCRIBE, DESC, EXPLAIN, and read-only WITH. Never submit an "
        "unfinished statement. Use mysql_list_tables and mysql_describe_table "
        "before guessing schema names."
    ),
)

POSTGRES_TOOL = FunctionTool.from_defaults(
    fn=postgres_query_tool,
    name="postgres_query",
    description=(
        "Execute exactly one complete read-only PostgreSQL statement in the "
        "`sql` argument."
    ),
)

MYSQL_LIST_TOOL = FunctionTool.from_defaults(
    fn=lambda: list_tables_tool("mysql"),
    name="mysql_list_tables",
    description=(
        "List tables in the configured MySQL database. This tool takes no " "arguments."
    ),
)

MYSQL_DESCRIBE_TABLE_TOOL = FunctionTool.from_defaults(
    fn=describe_mysql_table_tool,
    name="mysql_describe_table",
    description=(
        "Describe the columns of one MySQL table. Pass only the table name in "
        "`table_name`; do not pass SQL or a schema-qualified name."
    ),
)

MYSQL_RELATIONSHIPS_TOOL = FunctionTool.from_defaults(
    fn=mysql_relationships_tool,
    name="mysql_relationships",
    description=(
        "Return declared foreign-key relationships in the configured MySQL "
        "database. This tool takes no arguments."
    ),
)

POSTGRES_LIST_TOOL = FunctionTool.from_defaults(
    fn=lambda: list_tables_tool("postgres"),
    name="postgres_list_tables",
    description=(
        "List tables in the configured PostgreSQL database. This tool takes "
        "no arguments."
    ),
)

MYSQL_TOOLS = [
    MYSQL_TOOL,
    MYSQL_LIST_TOOL,
    MYSQL_DESCRIBE_TABLE_TOOL,
    MYSQL_RELATIONSHIPS_TOOL,
]

ADMIN_TOOLS = [
    POSTGRES_TOOL,
    POSTGRES_LIST_TOOL,
]
