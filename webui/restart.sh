#!/usr/bin/env bash
# Safe dashboard (re)start.
#
# 1. REFUSES to kill the server while a GPU job is running (pass --force to
#    override) -- killing mid-job orphans the subprocess on the GPU.
# 2. Launches fully detached (setsid + nohup + </dev/null) so the server
#    survives terminal/session teardown.
#
#   ./webui/restart.sh            # safe restart
#   ./webui/restart.sh --force    # restart even if a job is running
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${SA_HOST:-100.74.77.125}"
PORT="${SA_PORT:-8800}"

running=$(curl -s -m3 "http://$HOST:$PORT/api/state" 2>/dev/null \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["running"])' 2>/dev/null || echo "None")
if [ "$running" != "None" ] && [ "${1:-}" != "--force" ]; then
  echo "REFUSING: job $running is running (use --force to override)"; exit 1
fi

pkill -f "webui/app.py" 2>/dev/null || true
sleep 1

cd "$REPO"
export SA_PYTHON="${SA_PYTHON:-/home/dream/Desktop/AI/sdxl-travel-ablation/.venv/bin/python}"
export SA_FLUX2_MODEL="${SA_FLUX2_MODEL:-black-forest-labs/FLUX.2-klein-9B}"
export SA_HOST="$HOST" SA_PORT="$PORT"
setsid nohup "$SA_PYTHON" webui/app.py > /tmp/sa_webui.log 2>&1 < /dev/null &
sleep 2
if curl -s -m5 -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/" | grep -q 200; then
  echo "dashboard up: http://$HOST:$PORT (log: /tmp/sa_webui.log)"
else
  echo "starting... check /tmp/sa_webui.log"
fi
