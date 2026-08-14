"""Turning a directory of numbered frames into an .mp4 — torch-free.

Shared by every renderer that produces a sequence (``scripts/morph_film.py``'s
latent travel, ``scripts/animate.py``'s motion-module animation) so there is one
answer to "which ffmpeg, which encoder, which flags" rather than one per script.

The awkward part this exists for: a conda ffmpeg early on ``PATH`` is often built
``--disable-gpl`` and therefore has **no libx264 at all**, which is only
discovered after a full render has already been paid for. :func:`pick_h264_encoder`
probes every ffmpeg it can reach up front and falls back to another H.264 encoder
rather than dying at the end. ``SA_FFMPEG`` pins a specific binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def pick_h264_encoder(crf: int, log: str = "video") -> tuple[str, list[str]]:
    """Find an ffmpeg that can actually write H.264, and the flags to do it."""
    cands = [os.environ.get("SA_FFMPEG"), shutil.which("ffmpeg"),
             "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    seen, fallback = set(), None
    for c in cands:
        if not c or c in seen or not Path(c).is_file():
            continue
        seen.add(c)
        try:
            enc = subprocess.run([c, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:                                      # noqa: BLE001
            continue
        if "libx264" in enc:
            return c, ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
        if fallback is None and "libopenh264" in enc:
            # No CRF mode in openh264; ~12 Mbit keeps 512-1024px frames clean.
            fallback = (c, ["-c:v", "libopenh264", "-b:v", "12M"])
        if fallback is None and "h264_nvenc" in enc:
            fallback = (c, ["-c:v", "h264_nvenc", "-cq", str(crf)])
    if fallback:
        print(f"[{log}] no libx264 build found; encoding H.264 with "
              f"{fallback[1][1]} ({fallback[0]})", flush=True)
        return fallback
    raise SystemExit(
        f"[{log}] no ffmpeg with an H.264 encoder found (tried "
        f"{', '.join(sorted(seen)) or 'nothing on PATH'}). Install ffmpeg with "
        "libx264, or point SA_FFMPEG at one that has it.")


def encode_mp4(frame_glob: str, mp4: Path, fps: int, crf: int = 18,
               log: str = "video") -> Path:
    """Mux ``frame_%04d.ext`` into ``mp4``. Raises with ffmpeg's own tail on failure."""
    ffmpeg, codec_args = pick_h264_encoder(crf, log)
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", frame_glob,
           # yuv420p needs even dimensions; a refine scale can land on odd ones.
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           *codec_args, "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n".join(proc.stderr.strip().splitlines()[-15:]), flush=True)
        raise SystemExit(f"[{log}] ffmpeg failed (rc={proc.returncode})")
    return mp4


def smooth_mp4(mp4: Path, fps: int, crf: int = 18, log: str = "video") -> Path:
    """Motion-compensated frame interpolation -> ``<name>_smooth.mp4``.

    A 16-frame AnimateDiff clip is two seconds at its native 8fps and reads as
    choppy. ffmpeg's ``minterpolate`` estimates motion between the frames that
    exist and synthesises the ones between, which is honest smoothing -- unlike
    asking the model for more frames than it was trained on, which just produces
    noise.
    """
    ffmpeg, codec_args = pick_h264_encoder(crf, log)
    out = mp4.with_name(f"{mp4.stem}_smooth.mp4")
    cmd = [ffmpeg, "-y", "-i", str(mp4),
           "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
           *codec_args, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n".join(proc.stderr.strip().splitlines()[-10:]), flush=True)
        raise SystemExit(f"[{log}] ffmpeg minterpolate failed (rc={proc.returncode})")
    return out
