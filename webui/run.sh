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
# Prefer the repo's uv-managed .venv (torch 2.6 + diffusers 0.39 + transformers 5.x
# with the torchvision-free CLIP fallback). Fall back to whatever python is on PATH
# so an activated env still works if the .venv is absent.
if [[ -x "$REPO/.venv/bin/python" ]]; then
  DEFAULT_PY="$REPO/.venv/bin/python"
else
  DEFAULT_PY="$(command -v python3 || command -v python || true)"
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
export SA_SD15_CKPT="${SA_SD15_CKPT:-/home/rednax/SSD2TB/Github_repos/ComfyUI/models/checkpoints/SD15/juggernaut_reborn.safetensors}"

echo "[run] repo      = $REPO"
echo "[run] jobs py   = $SA_PYTHON"
echo "[run] bind      = http://$SA_HOST:$SA_PORT"
echo "[run] open from another tailnet device at the URL above."

# The server itself only needs fastapi+uvicorn (present in the venv). Run it
# with the same interpreter that runs the jobs for simplicity.
exec "$SA_PYTHON" "$REPO/webui/app.py"
