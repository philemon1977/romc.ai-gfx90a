#!/usr/bin/env bash
# Monitor a vLLM container: exits (and reports) on success OR failure keywords,
# instead of only waiting for /health. Usage: watch_container.sh <name> [timeout_s] [port]
NAME=${1:?container name}
TIMEOUT=${2:-1800}
PORT=${3:-8100}

SUCCESS_RE='Selected .*Int8|init engine|Started server process|Application startup complete'
FAIL_RE='RuntimeError|initialization failed|EngineCore failed|ValueError|TypeError|AssertionError|Traceback|failed to start|No available memory|exceeds available'

start=$(date +%s)
while :; do
  fail=$(docker logs "$NAME" 2>&1 | grep -E "$FAIL_RE" | tail -6)
  if [ -n "$fail" ]; then
    echo "### FAILURE DETECTED in $NAME ###"
    echo "$fail" | cut -c1-200
    echo "--- last 5 log lines ---"
    docker logs "$NAME" 2>&1 | tail -5 | cut -c1-200
    exit 1
  fi
  if curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "### HEALTHY: http://127.0.0.1:${PORT} ###"
    docker logs "$NAME" 2>&1 | grep -E "$SUCCESS_RE" | tail -4 | cut -c1-200
    exit 0
  fi
  if [ $(( $(date +%s) - start )) -gt "$TIMEOUT" ]; then
    echo "### TIMEOUT after ${TIMEOUT}s (no success, no failure keyword) ###"
    docker logs "$NAME" 2>&1 | tail -4 | cut -c1-200
    exit 2
  fi
  sleep 5
done
