#!/bin/bash
set -e

source /Users/jeffgatwood/multi_agent_workflow/.env

echo "=== Testing MySQL via JSON-RPC ==="
echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"mysql_query\",\"arguments\":{\"connectionString\":\"mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DATABASE}\",\"query\":\"SELECT 1 AS mysql_ok\"}}}" \
  | dbmcp

echo ""
echo "=== Testing Postgres via JSON-RPC ==="
echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"postgres_query\",\"arguments\":{\"connectionString\":\"postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DATABASE}\",\"query\":\"SELECT 1 AS postgres_ok\"}}}" \
  | dbmcp

echo ""
echo "=== All tests passed ==="

