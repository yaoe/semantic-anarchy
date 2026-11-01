"""ComfyUI executor: the torch-free half (name matching, prompt escaping, graphs)."""

import pytest

from semantic_anarchy.comfy import (
    ComfyError, encode_graph, escape_prompt, generate_graph, match_checkpoint,
)

LINUX = ["SD1.5/v1-5-pruned-emaonly.ckpt", "sdxl/base.safetensors"]
WINDOWS = ["SD1.5\\v1-5-pruned-emaonly.ckpt", "ltx-2.3-22b-dev.safetensors"]


def test_default_matches_across_path_separators():
    assert match_checkpoint(LINUX, "SD1.5/v1-5-pruned-emaonly.ckpt") == LINUX[0]
    assert match_checkpoint(WINDOWS, "SD1.5/v1-5-pruned-emaonly.ckpt") == WINDOWS[0]


def test_local_ckpt_hint_matches_by_file_name():
    hint = "/home/me/models/SD15/v1-5-pruned-emaonly.ckpt"
    assert match_checkpoint(WINDOWS, "missing.ckpt", hint=hint) == WINDOWS[0]


def test_unmatched_hint_falls_back_to_default():
    assert match_checkpoint(LINUX, LINUX[1], hint="/elsewhere/other.ckpt") == LINUX[1]


def test_missing_default_names_what_the_box_has():
    with pytest.raises(ComfyError, match="ltx-2.3"):
        match_checkpoint(WINDOWS, "SD1.5/juggernaut.safetensors")


def test_escape_prompt_keeps_weighting_syntax_literal():
    assert escape_prompt("a (red:1.3) fox") == "a \\(red:1.3\\) fox"


def test_encode_graph_one_output_per_prompt():
    g, outs = encode_graph("m.ckpt", ["a", "b", "c"])
    assert outs == ["b64_0", "b64_1", "b64_2"]
    assert sum(n["class_type"] == "CheckpointLoaderSimple" for n in g.values()) == 1
    assert all(g[o]["class_type"] == "ConditioningToBase64" for o in outs)


def test_generate_graph_branch_per_conditioning_with_own_seed():
    g, outs = generate_graph("m.ckpt", ["c0", "c1"], [7, 8], steps=20, cfg=7.5,
                             sampler="ddim", scheduler="ddim_uniform", width=768, height=512)
    assert outs == ["img0", "img1"]
    assert [g[f"ks{i}"]["inputs"]["seed"] for i in range(2)] == [7, 8]
    assert g["latent"]["inputs"] == {"width": 768, "height": 512, "batch_size": 1}
    assert g["neg"]["class_type"] == "CLIPTextEncode"          # empty-prompt negative
    g2, _ = generate_graph("m.ckpt", ["c0"], [1], steps=1, cfg=1.0, sampler="euler",
                           scheduler="normal", width=512, height=512, negative_b64="n")
    assert g2["neg"]["class_type"] == "Base64ToConditioning"
