"""
tools/mcp_tools.py
───────────────────
LlamaIndex FunctionTool wrappers that call persistent Database-MCP servers
(running via a background Popen lifecycle) for MySQL and Postgres queries.
These tools are handed to the ReAct agent inside each agent node.
"""

from __future__ import annotations
import json
import subprocess
import atexit
import traceback
from typing import Any, Dict
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings as LlamaSettings
from config.settings import settings as db_config

# ── Persistent MCP Connection Manager ─────────────────────────────────────────


class MCPConnectionManager:
    """
    Singleton Manager to orchestrate the lifecycle of persistent background
    MCP processes on macOS without blocking the agent runtime.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPConnectionManager, cls).__new__(cls)
            cls._instance.sessions = {}
            # Register an automatic shutdown hook to kill background daemons cleanly
            atexit.register(cls._instance.close_all_sessions)
        return cls._instance

    def get_session(self, db_type: str) -> subprocess.Popen:
        """Retrieves an existing persistent session or initializes a new one."""
        if db_type in self.sessions:
            # Check if the process is still alive
            if self.sessions[db_type].poll() is None:
                return self.sessions[db_type]
            print(f"⚠️ {db_type} MCP server disconnected unexpectedly. Restarting...")

        print(f"🚀 Launching persistent background MCP server for {db_type}...")

        # Build individual parameters using your secure .env config passed through settings
        if db_type == "postgres":
            args = [
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
        elif db_type == "mysql":
            args = [
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
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        # Spawn the long-lived daemon process asynchronously
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Completely unbuffered streams for instantaneous message delivery
        )

        # Execute the mandatory Model Context Protocol initialization handshake
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": f"LlamaIndex-{db_type}-Agent",
                    "version": "1.0.0",
                },
            },
        }

        try:
            # Write handshake payload to the stream
            process.stdin.write(json.dumps(init_payload) + "\n")
            process.stdin.flush()

            # Read and discard the immediate initialization confirmation line
            _ = process.stdout.readline()

            # Also send the required initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            process.stdin.write(json.dumps(initialized_notification) + "\n")
            process.stdin.flush()

            self.sessions[db_type] = process
            print(
                f"✅ {db_type.upper()} MCP server initialized and persistently locked open."
            )

        except Exception as e:
            process.terminate()
            raise RuntimeError(
                f"Failed to complete handshake with {db_type} MCP server: {e}"
            )

        return process

    def close_all_sessions(self):
        """Gracefully closes all background daemons."""
        for db_type, process in list(self.sessions.items()):
            if process.poll() is None:
                print(f"🧹 Closing persistent {db_type} MCP background process...")
                process.terminate()
                process.wait()
        self.sessions.clear()


# Instantiate our unified persistent supervisor
mcp_manager = MCPConnectionManager()


# ── Synchronous Communication Wrapper ─────────────────────────────────────────
def _call_persistent_mcp(db_type: str, method: str, params: Dict[str, Any]) -> Any:
    """
    Communicates via standard JSON-RPC over the persistent background streams
    associated with the chosen database target.
    """
    process = mcp_manager.get_session(db_type)

    payload = {"jsonrpc": "2.0", "id": 2, "method": method, "params": params}

    try:
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError(
                f"The {db_type} MCP server closed the communication stream."
            )

        response = json.loads(response_line)

        if "error" in response:
            raise RuntimeError(f"MCP server returned error: {response['error']}")

        return response.get("result")

    except Exception as e:
        if db_type in mcp_manager.sessions:
            del mcp_manager.sessions[db_type]
        raise RuntimeError(
            f"Failed to communicate with persistent {db_type} MCP server: {e}"
        )


# ── MySQL Tool Execution Function ─────────────────────────────────────────────
def mysql_query_tool(sql: str) -> str:
    """Execute a SQL query against the MySQL database."""
    try:
        result = _call_persistent_mcp(
            db_type="mysql",
            method="tools/call",
            params={
                "name": "readQuery",
                "arguments": {
                    "query": sql  # FIX: dbmcp's "readQuery" expects "query", not "sql"
                },
            },
        )

        if isinstance(result, dict) and result.get("isError") is True:
            content_list = result.get("content", [])
            if isinstance(content_list, list) and len(content_list) > 0:
                first_item = content_list[0]
                if isinstance(first_item, dict):
                    return (
                        f"Database Error: {first_item.get('text', 'Malformed block')}"
                    )
            return f"Database Error: {str(content_list)}"

        content = result.get("content", []) if isinstance(result, dict) else result
        if not content or content == [] or content == "[]":
            return "Query executed successfully. No rows or tables were returned."

        return json.dumps(content, indent=2, default=str)

    except Exception as e:
        print("\n" + "=" * 50)
        print("!!! CRITICAL MCP WRAPPER ERROR IN MYSQL !!!")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()
        print("=" * 50 + "\n", flush=True)
        return f"Python Wrapper Exception caught: {str(e)}"


# ── Postgres Tool Execution Function ──────────────────────────────────────────
def postgres_query_tool(sql: str) -> str:
    """Execute a SQL query against the Postgres vector DB."""
    try:
        result = _call_persistent_mcp(
            db_type="postgres",
            method="tools/call",
            params={
                "name": "readQuery",
                "arguments": {
                    "query": sql  # FIX: dbmcp's "readQuery" expects "query", not "sql"
                },
            },
        )

        if isinstance(result, dict) and result.get("isError") is True:
            content_list = result.get("content", [])
            if isinstance(content_list, list) and len(content_list) > 0:
                first_item = content_list[0]
                if isinstance(first_item, dict):
                    return (
                        f"Database Error: {first_item.get('text', 'Malformed block')}"
                    )
            return f"Database Error: {str(content_list)}"

        content = result.get("content", []) if isinstance(result, dict) else result
        if not content or content == [] or content == "[]":
            return "Query executed successfully. No rows or tables were returned."

        return json.dumps(content, indent=2, default=str)

    except Exception as e:
        print("\n" + "=" * 50)
        print("!!! CRITICAL MCP WRAPPER ERROR IN POSTGRES !!!")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()
        print("=" * 50 + "\n", flush=True)
        return f"Python Wrapper Exception caught: {str(e)}"


# ── Dynamic Native Schema Extractor (No Raw SQL Required) ─────────────────────
def list_tables_tool(db_type: str) -> str:
    """Retrieve all available table structures directly using dbmcp's native capability."""
    try:
        result = _call_persistent_mcp(
            db_type=db_type,
            method="tools/call",
            params={"name": "listTables", "arguments": {}},
        )
        content = result.get("content", []) if isinstance(result, dict) else result
        return json.dumps(content, indent=2, default=str)
    except Exception as e:
        return f"Failed to natively list tables for {db_type}: {str(e)}"


def describe_mysql_table_tool(table_name: str) -> str:
    """
    Safely describe a MySQL table's columns without allowing arbitrary SQL.
    """
    try:
        safe_table = str(table_name).strip()

        # Basic identifier guard
        if not safe_table.replace("_", "").isalnum():
            return f"Invalid table name: {safe_table}"

        sql = f"DESCRIBE `{safe_table}`"

        result = _call_persistent_mcp(
            db_type="mysql",
            method="tools/call",
            params={
                "name": "readQuery",
                "arguments": {"query": sql},
            },
        )

        content = result.get("content", []) if isinstance(result, dict) else result

        if not content or content == [] or content == "[]":
            return f"No column information returned for table `{safe_table}`."

        return json.dumps(content, indent=2, default=str)

    except Exception as e:
        return f"Failed to describe MySQL table `{table_name}`: {str(e)}"


# LamaIndex FunctionTool wrappers
MYSQL_TOOL = FunctionTool.from_defaults(
    fn=mysql_query_tool,
    name="mysql_query",
    description="""
    Query the MySQL database.

    IMPORTANT:
    - Never assume table names or columns.
    - Use mysql_list_tables first if schema is uncertain.
    - If a query fails due to a missing table or column,
      inspect the schema before trying again.
    - Do not repeatedly guess names.
    """,
)

POSTGRES_TOOL = FunctionTool.from_defaults(
    fn=postgres_query_tool,
    name="postgres_query",
    description="Query the Postgres database directly. Input: valid PostgreSQL SELECT statement.",
)

MYSQL_LIST_TOOL = FunctionTool.from_defaults(
    fn=lambda: list_tables_tool("mysql"),
    name="mysql_list_tables",
    description="Lists all tables natively inside the connected MySQL database without running a manual query string.",
)

MYSQL_DESCRIBE_TABLE_TOOL = FunctionTool.from_defaults(
    fn=describe_mysql_table_tool,
    name="mysql_describe_table",
    description=(
        "Safely describes the columns of one MySQL table. "
        "Input: table_name only. Do not pass SQL."
    ),
)

POSTGRES_LIST_TOOL = FunctionTool.from_defaults(
    fn=lambda: list_tables_tool("postgres"),
    name="postgres_list_tables",
    description="Lists all tables natively inside the connected Postgres database without running a manual query string.",
)

MYSQL_TOOLS = [
    MYSQL_TOOL,
    MYSQL_LIST_TOOL,
    MYSQL_DESCRIBE_TABLE_TOOL,
]

ADMIN_TOOLS = [
    POSTGRES_TOOL,
    POSTGRES_LIST_TOOL,
]
