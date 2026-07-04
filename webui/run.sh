#!/usr/bin/env bash
# Launch the Semantic Anarchy explorer dashboard, bound to this machine's
# Tailscale IP so it's reachable from other devices on the tailnet — but NOT
# exposed on the local LAN / public interfaces.
#
#   ./webui/run.sh                # auto: tailscale IP, venv python, port 8800
#   SA_PORT=9000 ./webui/run.sh   # override port
#   SA_HOST=0.0.0.0 ./webui/run.sh# bind everything (LAN too) if you really want
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Interpreter that has torch+diffusers (jobs run via this). Override with SA_PYTHON.
DEFAULT_PY="/home/dream/Desktop/AI/sdxl-travel-ablation/.venv/bin/python"
export SA_PYTHON="${SA_PYTHON:-$DEFAULT_PY}"

# Bind host: Tailscale IP by default (tailnet-only exposure).
if [[ -z "${SA_HOST:-}" ]]; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  export SA_HOST="${TS_IP:-127.0.0.1}"
fi
export SA_PORT="${SA_PORT:-8800}"

echo "[run] repo      = $REPO"
echo "[run] jobs py   = $SA_PYTHON"
echo "[run] bind      = http://$SA_HOST:$SA_PORT"
echo "[run] open from another tailnet device at the URL above."

# The server itself only needs fastapi+uvicorn (present in the venv). Run it
# with the same interpreter that runs the jobs for simplicity.
exec "$SA_PYTHON" "$REPO/webui/app.py"
