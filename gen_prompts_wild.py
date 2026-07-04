#!/usr/bin/env python3
"""Generate a WILD but VARIED corpus -- strange across many moods, not macabre.

The point is a high-variance, surprising embedding cloud (so samples come out
wild instead of mild), but spread across registers -- whimsical, absurd, cosmic,
biomechanical, candy-pop, folkloric, dreamlike, with only a pinch of dark -- so
the territory is *strange and wondrous*, not a death-metal monoculture.

Each prompt still names concrete things (the UNet needs something to render) but
combines them the way no prompter would -> a wide, weird cloud.

    python gen_prompts_wild.py --n 1000 --seed 7 --out prompts_1000.txt
"""

from __future__ import annotations

import argparse
import os
import random

ENTITIES = [
    # whimsical creatures
    "a jellyfish cathedral", "a snail the size of a lighthouse", "a paper-crane dragon",
    "a librarian octopus", "a king of mushrooms", "a moth with cathedral-window wings",
    "a tortoise carrying a garden", "a fox made of autumn", "a whale of constellations",
    "a hummingbird wearing a crown", "a parade-balloon deity", "a choir of lanterns",
    # absurd characters
    "a cat-headed accountant", "a robot monk", "a tiny astronaut", "a disco golem",
    "a grandmother made of clouds", "a postman of the deep sea", "a clockwork ballerina",
    "a beekeeper of stars", "a gardener of glass flowers", "a knight of folded paper",
    "a wizard of vending machines", "a sentient quilt", "a diver in a tuxedo",
    # cosmic / mythic
    "a sleeping star-giant", "a many-armed dancing goddess", "a phoenix of stained glass",
    "a leviathan of warm light", "a spirit of the northern lights", "a titan of coral",
    "a fox-spirit with paper-lantern tails", "an angel of radio waves",
    # biomechanical / techno-organic (strange, not gory)
    "a cathedral-sized music box", "a city that breathes", "a garden of clockwork insects",
    "a violin growing roots", "a lighthouse made of circuitry", "a train of living vines",
    "a telephone switchboard forest", "a chandelier jellyfish",
    # ordinary things, transfigured
    "a teapot pouring galaxies", "a staircase that loves the sky", "a library that is an ocean",
    "an umbrella sheltering a thunderstorm", "a chair dreaming of being a tree",
    "a bicycle made of rivers", "a piano blooming with koi",
]
MATERIALS = [
    "made of stained glass", "of origami and gold leaf", "of bubblegum and chrome",
    "of moss and neon", "woven from light", "of soap bubbles", "of knitted wool",
    "of candy and clockwork", "of porcelain and ferns", "of paper lanterns",
    "of liquid mercury and petals", "of crystal and smoke", "of stitched maps",
    "of woven river-water", "of glazed ceramic", "of iridescent beetle shells",
    "of folded silk", "of frosted sugar glass", "of luminous fungus",
]
STATES = [
    "mid-pirouette", "unfolding into a garden", "sneezing constellations",
    "blooming with lanterns", "juggling small moons", "dissolving into butterflies",
    "growing a tiny forest", "conducting an orchestra of rain", "drifting upward",
    "turning into a flock of birds", "spilling a river of light", "humming a melody you can see",
    "sprouting wings of paper", "caught mid-transformation", "exhaling fireflies",
    "balancing a galaxy on one finger", "weaving itself from smoke", "blossoming open",
]
SETTINGS = [
    "in a floating night market", "on a carousel at the edge of space",
    "inside a giant music box", "in a greenhouse on the moon", "on a checkerboard desert",
    "in a library that is also an ocean", "under twin suns", "in a city of soap bubbles",
    "on a lake of mirrors", "in a forest of paper lanterns", "in a cathedral of coral",
    "at a tea party in the clouds", "in a canyon of stained glass", "on a beach of stars",
    "inside a snow globe at dusk", "in a meadow that floats", "in a desert of folded silk",
]
STYLES = [
    "Studio Ghibli", "Moebius ligne claire", "James Jean", "Takashi Murakami superflat pop",
    "storybook gouache", "claymation stop-motion", "felt diorama macro", "risograph print",
    "vaporwave", "psychedelic 70s poster", "Rococo excess", "ukiyo-e woodblock",
    "Alphonse Mucha art nouveau", "blacklight fluorescent", "tilt-shift miniature",
    "stained-glass mosaic", "Beksinski dream-oil", "Hieronymus Bosch detail",
    "watercolor and ink", "iridescent 3D render", "papercraft pop-up book", "Kandinsky abstraction",
]
EXTRA = [
    "iridescent", "soft golden light", "dreamy bokeh", "saturated jewel tones",
    "warm rim light", "fish-eye whimsy", "intricate detail", "luminous glow",
    "pastel palette", "high-key lighting", "macro close-up", "shallow depth of field",
    "candy colors", "volumetric haze", "playful composition", "cinematic wide shot",
]


def build(rng: random.Random) -> str:
    parts = [rng.choice(ENTITIES)]
    if rng.random() < 0.8:
        parts[0] += " " + rng.choice(MATERIALS)
    if rng.random() < 0.85:
        parts.append(rng.choice(STATES))
    if rng.random() < 0.8:
        parts.append(rng.choice(SETTINGS))
    # occasionally fuse a SECOND entity for surprising juxtaposition
    if rng.random() < 0.3:
        parts.append("fused with " + rng.choice(ENTITIES))
    parts.append(rng.choice(STYLES))
    for _ in range(rng.randint(1, 2)):
        parts.append(rng.choice(EXTRA))
    return ", ".join(parts)


def generate(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen, out, attempts = set(), [], 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        p = build(rng)
        if p not in seen:
            seen.add(p)
            out.append(p)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} unique; enlarge vocab or lower --n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "prompts_1000.txt"))
    args = ap.parse_args(argv)
    prompts = generate(args.n, args.seed)
    with open(args.out, "w") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"[gen_prompts_wild] wrote {len(prompts)} wild-but-varied prompts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
