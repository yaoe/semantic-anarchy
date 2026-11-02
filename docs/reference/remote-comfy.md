# Remote ComfyUI execution and job lanes

sd15 can render on any ComfyUI box instead of in-process diffusers, and the dashboard routes per job. Read `semantic_anarchy/comfy.py` (HTTP client, graph builders, unpickler), `ComfySD15Backend` in `backend.py`, and the "ComfyUI servers" and `Runner` sections of `webui/app.py`.

## Executor

With `SA_COMFY_URL` set, `make_backend("sd15")` returns `ComfySD15Backend`: `encode` runs `CheckpointLoaderSimple → CLIPTextEncode → ConditioningToBase64` and unpickles the result out of `/history`; `generate` sends each sampled tensor in as a `Base64ToConditioning` pickle → `KSampler` → `PreviewImage` and fetches the PNG via `/view`. Both nodes come from the RES4LYF pack (its unpickler only admits torch tensor classes). Mine through the same box you generate on, so the distribution lives in Comfy's encoder space. `SA_COMFY_SD15_CKPT` names the checkpoint as Comfy lists it; `--scheduler` maps through `SAMPLERS`. Scripts swap schedulers via `backend.set_scheduler`, never `set_scheduler(backend.model.pipe, …)` — the Comfy backend has no pipe. img2img/IP-Adapter, refine, upscale and films are still diffusers-only: any script that touches `backend.model` loads with `load_backend(args, remote_ok=False)` / `make_backend(..., remote_ok=False)` so `SA_COMFY_URL` can never hand it the model-less Comfy backend (a test greps `scripts/` for this), and `app.py` submits those jobs with `job_env(None)`. Payloads coming back from a box are unpickled through `comfy._CondUnpickler` (torch tensor rebuild + `OrderedDict` only; storage bytes via `torch.load(weights_only=True)`) — never a bare `pickle.loads`. Sidecars record `executor` (`diffusers`/`comfy`); upscale and films warn when they replay a Comfy-made image, since its noise and sampler differ. A Windows box lists checkpoints with backslashes (`SD1.5\v1-5-pruned-emaonly.ckpt`); `ComfySD15Backend.load` matches names slash-insensitively (`comfy.match_checkpoint`). Machine-specific rules about *which* boxes may receive jobs belong in a gitignored `CLAUDE.local.md`, not here.

**Negative prompt.** `ComfySD15Backend.generate` honours the same four `--neg-mode`s as `SD15Backend._negative`: `text` encodes `negative_prompt` on the box (the house default or `--negative`, parentheses escaped so Comfy's weighting syntax stays literal), `empty`, `zeros`, and `mean` sent as a `Base64ToConditioning` tensor. Holder lookup is `cli_args._negative_holder`, because this backend has no `.model`.

**No length dimension.** There is no tokenizer on this side, so `token_lengths` is None and a Comfy-mined fit has no content/padding split: `--length-mode` on it is reported as ignored.

## Routing

The dashboard can send each sd15 job to a different box. `comfy_servers()` in `app.py` is the allow-list: `SA_COMFY_SERVERS="gpu1=https://…,gpu2=https://…"`, else the gitignored `webui/comfy_servers.json` (`{"servers": {"gpu1": "https://…"}}`), plus `SA_COMFY_URL` as the default. A request can only name a key from that list — never a URL. `RunRequest.server` (also on `ExploreRequest`/`EvolveRequest`, which the frontend fills from the sidebar's *Run on* field) is a key from that list or `local`; `resolve_server` turns it into a lane plus a URL, and the URL reaches the script only as `SA_COMFY_URL` in that job's env (`job_env`). Non-sd15 backends always run `local`. `GET /api/servers` probes each box read-only (reachable, SD1.5 listed, ComfyUI-Manager download still in flight, queue depth; cached 10 s) for the picker's ●/◐/○ status.

A job is local unless it names a box: `Runner.submit` with no `env` unsets `SA_COMFY_URL`, so new endpoints can't inherit the process-wide default by accident. `/api/run` with init images forces the local lane, since img2img and IP-Adapter need the pipeline.

**Trap: distributions aren't keyed by box.** A fit is named after the *local* checkpoint selection. Mining with a box selected writes to that same file, replacing the local encoder's fit with the box's, and generating on a box samples whatever fit is selected.

## Lanes

One worker thread per lane, one job at a time per lane. `exclusive=True` jobs (`mine`, including `/api/dist/encode`, and `evolve`) wait for every lane to go idle and hold the rest off. `/api/state.running` stays a single id for `restart.sh`; `running_ids` lists all of them.
