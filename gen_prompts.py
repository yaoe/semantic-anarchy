#!/usr/bin/env python3
"""Deterministic generator for a wide, "good" prompt corpus.

Produces N (default 1000) UNIQUE, diverse, natural-language prompts by combining
~4-6 components drawn from rich themed lists — broad enough to index most of what
SDXL can encode (the deck's "1000 good prompts"). Seeded, so re-running gives the
same corpus. Phrases read as coherent descriptions, not comma-spam.

    python gen_prompts.py                       # -> prompts_1000.txt (1000 lines)
    python gen_prompts.py --n 1000 --seed 7 --out prompts_1000.txt
"""

from __future__ import annotations

import argparse
import os
import random


# --- component vocabularies ------------------------------------------------

SUBJECTS = [
    # people / portraits
    "a weathered old fisherman", "a young ballerina mid-leap", "a masked carnival dancer",
    "a stoic samurai at rest", "a street violinist", "a queen in ceremonial robes",
    "twin children playing", "an astronaut removing her helmet", "a blacksmith at the forge",
    "a lone monk crossing a courtyard",
    # animals
    "a snow leopard on a ridge", "a murmuration of starlings", "a sleeping arctic fox",
    "a humpback whale breaching", "a peacock displaying", "a pack of wolves in fog",
    "a hummingbird at a flower", "a herd of elephants at dusk", "a barn owl in flight",
    "a koi pond seen from above",
    # mythical creatures
    "a coiled emerald dragon", "a phoenix rising from embers", "a moonlit kelpie",
    "a stone golem awakening", "a griffin perched on a spire", "a sea serpent surfacing",
    "a forest spirit made of moss", "a many-eyed seraph",
    # objects / artifacts
    "an ornate pocket watch", "a cracked porcelain mask", "a glowing alchemical flask",
    "a ship in a bottle", "an antique brass telescope", "a tarot card spread",
    "a music box mid-melody", "a ring of ancient keys",
    # landscapes
    "a terraced rice valley at sunrise", "a frozen waterfall", "a field of lavender in wind",
    "a desert of red dunes", "a misty fjord", "a volcanic black-sand beach",
    "a canyon river bend", "an alpine meadow after rain",
    # architecture
    "a half-ruined cathedral", "a floating temple in the clouds", "a brutalist concrete library",
    "a winding medina street", "a glass skyscraper at twilight", "a lighthouse on a cliff",
    "an overgrown abandoned station", "a spiral stone staircase",
    # vehicles
    "a vintage steam locomotive", "a wooden sailing galleon", "a chrome hot rod",
    "a weathered fishing trawler", "a hot-air balloon over hills", "a derelict submarine",
    # plants / botanical
    "a single luminous mushroom", "a tangle of bioluminescent vines", "a bonsai in bloom",
    "a carnivorous plant unfurling", "a wreath of autumn leaves",
    # food
    "a stack of dripping pancakes", "a still life of split pomegranates",
    "a bowl of steaming ramen", "a tiered patisserie cake", "a market stall of spices",
    # sci-fi
    "a derelict orbital ringworld", "a robot tending a garden", "a cybernetic street vendor",
    "a colony dome on a red planet", "a swarm of repair drones",
    # abstract
    "an impossible Escher staircase", "a fractal of unfolding shapes",
    "a ribbon of liquid light", "interlocking geometric solids",
    # interiors
    "a sunlit artist's studio", "a cluttered apothecary", "a grand library reading room",
    "a cramped ramen bar at night", "a greenhouse full of ferns",
    # underwater
    "a coral reef teeming with fish", "a diver among jellyfish", "a sunken statue garden",
    # space
    "a nebula cradling newborn stars", "a ringed gas giant from its moon",
    "an eclipse over a barren world",
]

MEDIUMS = [
    "oil painting", "watercolor", "gouache", "3D render", "Octane render",
    "photograph", "cinematic film still", "concept art", "digital illustration",
    "ink drawing", "charcoal sketch", "pixel art", "paper collage", "risograph print",
    "matte painting", "pencil study",
]

AESTHETICS = [
    "impressionist", "surrealist", "baroque", "art nouveau", "brutalist",
    "vaporwave", "ukiyo-e", "art deco", "minimalist", "gothic", "cyberpunk",
    "romanticist", "bauhaus", "folk-art", "psychedelic",
]

LIGHTING = [
    "golden-hour light", "soft studio softbox lighting", "dramatic rim lighting",
    "volumetric god rays", "neon glow", "warm candlelight", "flat overcast light",
    "moody chiaroscuro", "cool moonlight", "harsh midday sun", "backlit silhouette",
    "bioluminescent glow",
]

MOODS = [
    "serene", "foreboding", "melancholic", "whimsical", "triumphant",
    "dreamlike", "tense", "nostalgic", "ecstatic", "solemn", "playful", "eerie",
]

COMPOSITIONS = [
    "extreme close-up", "wide establishing shot", "low-angle hero shot",
    "bird's-eye view", "symmetrical centered framing", "rule-of-thirds composition",
    "Dutch-angle framing", "shallow depth of field", "long telephoto compression",
    "fisheye perspective",
]

PALETTES = [
    "a muted earth-tone palette", "vivid complementary colors",
    "a cold monochrome blue palette", "warm amber and teal tones",
    "pastel candy colors", "high-contrast black and white",
    "iridescent pearlescent hues", "a limited two-color palette",
    "rich jewel tones", "washed-out faded colors",
]

QUALITY = [
    "highly detailed", "intricate textures", "sharp focus", "8k", "fine grain",
    "crisp linework", "painterly brushwork", "soft bokeh",
]


def build_prompt(rng: random.Random) -> str:
    """Compose one coherent natural phrase from ~4-6 components."""
    subject = rng.choice(SUBJECTS)
    medium = rng.choice(MEDIUMS)
    parts = [f"{subject}, {medium}"]

    # Optionally an aesthetic flavor (most of the time).
    if rng.random() < 0.8:
        parts.append(f"in a {rng.choice(AESTHETICS)} style")
    # Lighting (most of the time).
    if rng.random() < 0.85:
        parts.append(rng.choice(LIGHTING))
    # Composition (sometimes).
    if rng.random() < 0.6:
        parts.append(rng.choice(COMPOSITIONS))
    # Mood (sometimes).
    if rng.random() < 0.6:
        parts.append(f"{rng.choice(MOODS)} mood")
    # Palette (sometimes).
    if rng.random() < 0.5:
        parts.append(rng.choice(PALETTES))
    # A couple of quality tags (light touch).
    if rng.random() < 0.7:
        q = rng.sample(QUALITY, k=rng.randint(1, 2))
        parts.extend(q)

    return ", ".join(parts)


def generate(n: int, seed: int) -> list[str]:
    """Generate n UNIQUE prompts deterministically."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    # Cap attempts so a too-small vocabulary can't loop forever.
    max_attempts = n * 200
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        p = build_prompt(rng)
        if p not in seen:
            seen.add(p)
            out.append(p)
    if len(out) < n:
        raise RuntimeError(
            f"only produced {len(out)} unique prompts in {attempts} attempts; "
            f"enlarge the vocabularies or lower --n.")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--out", default=os.path.join(here, "prompts_1000.txt"))
    args = p.parse_args(argv)

    prompts = generate(args.n, args.seed)
    with open(args.out, "w") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"[gen_prompts] wrote {len(prompts)} unique prompts (seed={args.seed}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
