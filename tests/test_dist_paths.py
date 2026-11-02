"""Torch-free tests for the distribution-file naming rules (dist_paths.py).

These are what let the dashboard tell "already encoded" from "needs encoding"
without ever loading a model, so they have to agree exactly with what
``Backend.save_dists`` + ``cli_args.dist_prefix`` actually write.
"""

import json

import numpy as np
import pytest

from semantic_anarchy.backend import dist_backend
from semantic_anarchy.cli_args import dist_prefix
from semantic_anarchy.dist_paths import (
    backend_prefix, base_from_npz, dist_files, dist_meta, dist_ready,
    model_slug, prompt_dist_base, tensor_suffixes,
)
from semantic_anarchy.distribution import EmbeddingDistribution

BACKENDS = ["sd15", "sd2", "sdxl", "flux2", "krea2"]


class _Args:
    """The one attribute cli_args.dist_prefix reads."""

    def __init__(self, backend):
        self.backend = backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_prefix_matches_cli(backend):
    """The dashboard's rule IS the CLI's rule -- one drifting would point the
    two halves at different files."""
    assert backend_prefix("outputs/dist", backend) == dist_prefix(_Args(backend), "outputs/dist")


def test_sd15_keeps_the_bare_prefix():
    """The original outputs/dist layout survives; the others namespace."""
    assert backend_prefix("outputs/dist", "sd15") == "outputs/dist"
    assert backend_prefix("outputs/dist", "sdxl") == "outputs/dist_sdxl"


@pytest.mark.parametrize("backend", BACKENDS)
def test_dist_files_matches_what_save_writes(backend, tmp_path):
    """Fit -> save -> the files dist_files() predicted are exactly the ones there."""
    b = dist_backend(backend)
    base = str(tmp_path / "corpus__ckpt")
    rng = np.random.default_rng(0)
    dists = {
        name: EmbeddingDistribution.fit(rng.normal(size=(8, 5)))
        for name in b.tensor_names
    }
    written = b.save_dists(dists, backend_prefix(base, backend))
    assert sorted(map(str, written)) == sorted(str(p) for p in dist_files(base, backend))
    assert dist_ready(base, backend)


def test_dist_ready_is_false_until_every_tensor_lands(tmp_path):
    """A half-written sdxl fit is not usable -- both tensors or nothing."""
    base = str(tmp_path / "corpus")
    assert not dist_ready(base, "sdxl")
    files = dist_files(base, "sdxl")
    assert len(files) == 2
    files[0].write_bytes(b"")
    assert not dist_ready(base, "sdxl")
    files[1].write_bytes(b"")
    assert dist_ready(base, "sdxl")


def test_tensor_suffixes_follow_the_backend():
    assert tensor_suffixes("sd15") == ("",)
    assert tensor_suffixes("sdxl") == ("__prompt_embeds", "__pooled")


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_from_npz_round_trips(backend):
    """Any one .npz of a fit identifies the base the whole set belongs to."""
    base = "/data/corpora/xander__juggernaut"
    for f in dist_files(base, backend):
        assert base_from_npz(f, backend) == base
        assert base_from_npz(str(f), backend) == base


def test_model_slug_keeps_checkpoints_apart():
    """The slug is what makes one corpus hold several fits, so near-miss
    checkpoint names must not collapse onto each other."""
    assert model_slug("/models/v1-5-pruned-emaonly.safetensors") == "v1-5-pruned-emaonly"
    assert model_slug("/models/SD15/juggernaut_reborn.safetensors") == "juggernaut_reborn"
    # a diffusers folder is named, not stemmed
    assert model_slug("/models/sdxl-turbo/") == "sdxl-turbo"
    # an HF repo id keeps its org -- that is what disambiguates the forks
    assert model_slug("stabilityai/stable-diffusion-xl-base-1.0") == \
        "stabilityai-stable-diffusion-xl-base-1.0"
    assert model_slug("org/x") != model_slug("other/x")
    # always filename-safe
    assert "/" not in model_slug("weird name/with spaces!.safetensors")


def test_prompt_dist_base_sits_beside_the_corpus(tmp_path):
    """Latents travel with the .txt, tagged by the checkpoint that made them."""
    txt = tmp_path / "xander_prompts.txt"
    a = prompt_dist_base(txt, "/models/juggernaut_reborn.safetensors")
    b = prompt_dist_base(txt, "/models/v1-5-pruned-emaonly.safetensors")
    assert a.startswith(str(tmp_path))
    assert a.endswith("xander_prompts__juggernaut_reborn")
    assert a != b                       # two checkpoints -> two fits, no clobber
    # and each resolves back to a real filename under its backend
    assert dist_files(a, "sd15")[0].name == "xander_prompts__juggernaut_reborn.npz"


def test_dist_meta_reads_the_sidecar(tmp_path):
    base = str(tmp_path / "corpus")
    assert dist_meta(base, "sd15") is None            # nothing written yet
    dist_files(base, "sd15")[0].write_bytes(b"")
    assert dist_meta(base, "sd15") is None            # .npz but no sidecar
    (tmp_path / "corpus.meta.json").write_text(json.dumps({"n_samples": 4144}))
    assert dist_meta(base, "sd15")["n_samples"] == 4144
