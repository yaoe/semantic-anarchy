#!/usr/bin/env bash
# Launch the Semantic Anarchy explorer dashboard, bound to this machine's
# Tailscale IP so it's reachable from other devices on the tailnet — but NOT
# exposed on the local LAN / public interfaces.
#
#   ./webui/run.sh                # auto: tailscale IP, active python, port 8800
#   SA_PORT=9000 ./webui/run.sh   # override port
#   SA_HOST=0.0.0.0 ./webui/run.sh# bind everything (LAN too) if you really want
#   SA_PYTHON=/path/to/python ./webui/run.sh  # override the interpreter
#   SA_DISPLAY=:1 ./webui/run.sh  # X display for the sidebar's OS file dialog,
#                                 # if this shell is headless (else it inherits
#                                 # DISPLAY; without either, the picker falls
#                                 # back to the in-page file browser)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Interpreter that has torch+diffusers (jobs run via this). Override with SA_PYTHON.
# Prefer a repo-local .venv if one is ever built here; otherwise use the shared
# env this machine actually runs on (torch 2.11+cu128, diffusers 0.33.1) -- the
# same default restart.sh hardcodes, so the two scripts agree. NOT a bare PATH
# python3: that one has no fastapi, and app.py died on `import fastapi` several
# lines after this script printed "bind = ...", which reads like a network fault.
SHARED_PY="/home/dream/Desktop/AI/sdxl-travel-ablation/.venv/bin/python"
if [[ -x "$REPO/.venv/bin/python" ]]; then
  DEFAULT_PY="$REPO/.venv/bin/python"
else
  DEFAULT_PY="$SHARED_PY"
fi
export SA_PYTHON="${SA_PYTHON:-$DEFAULT_PY}"

if [[ -z "${SA_PYTHON:-}" || ! -x "$SA_PYTHON" ]]; then
  echo "[run] ERROR: no usable python found." >&2
  echo "[run] Activate the env with torch+diffusers, or set SA_PYTHON=/path/to/python." >&2
  exit 1
fi

# Bind host: Tailscale IP by default (tailnet-only exposure).
if [[ -z "${SA_HOST:-}" ]]; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  export SA_HOST="${TS_IP:-127.0.0.1}"
fi
export SA_PORT="${SA_PORT:-8800}"

# SD1.5 single-file checkpoint. Default points at a checkpoint that exists on
# this machine; override with SA_SD15_CKPT to use a different one. A checkpoint
# hand-picked in the dashboard's Model panel (webui/model_config.json) wins over
# this — this is only the fallback for a backend nothing has been picked for.
export SA_SD15_CKPT="${SA_SD15_CKPT:-$HOME/models/v1-5-pruned-emaonly.safetensors}"
export SA_SD2_CKPT="${SA_SD2_CKPT:-$HOME/models/v2-1_768-ema-pruned.safetensors}"

echo "[run] repo      = $REPO"
echo "[run] jobs py   = $SA_PYTHON"
echo "[run] bind      = http://$SA_HOST:$SA_PORT"
echo "[run] open from another tailnet device at the URL above."

# The server itself only needs fastapi+uvicorn (present in the venv). Run it
# with the same interpreter that runs the jobs for simplicity.
exec "$SA_PYTHON" "$REPO/webui/app.py"
