#!/usr/bin/env python3
"""Build a 1000-prompt HIGH-QUALITY 'good prompts' corpus (the quality anchor).

Seeds with the hand-curated prompts in prompts_good100.txt, then fills to N with
coherent, natural, DIVERSE-but-normal prompts combined from rich themed vocab
(broad subject/medium/style/lighting coverage). NOT weird -- wildness is meant to
come from sampling OUTSIDE the distribution these define, not from the prompts.

    python gen_prompts_good.py --n 1000 --seed 7 --out prompts_1000.txt
"""

from __future__ import annotations

import argparse
import os
import random

SUBJECTS = [
    # people / portraits
    "a weathered old fisherman", "a young ballerina mid-leap", "a masked carnival dancer",
    "a stoic samurai at rest", "a street violinist", "a queen in ceremonial robes",
    "an astronaut removing her helmet", "a blacksmith at the forge", "a lone monk in a courtyard",
    "an elderly couple dancing", "a ballet dancer stretching", "a jazz singer at a microphone",
    "a desert nomad in flowing robes", "a Renaissance noblewoman", "a weary traveller at a station",
    "a child blowing dandelion seeds", "a fortune teller with tarot cards", "a lighthouse keeper",
    "a beekeeper among the hives", "a potter at the wheel", "a falconer with a hooded hawk",
    "a deep-sea diver in a vintage suit", "a violin maker in his workshop", "a tea master pouring",
    # animals
    "a snow leopard on a ridge", "a murmuration of starlings", "a sleeping arctic fox",
    "a humpback whale breaching", "a peacock displaying", "a pack of wolves in fog",
    "a hummingbird at a flower", "a herd of elephants at dusk", "a barn owl in flight",
    "a tiger drinking at a river", "a stag in misty highlands", "a heron at first light",
    "a horse galloping through surf", "a chameleon on a branch", "a polar bear on drifting ice",
    "a fox hunting in deep snow", "a kingfisher diving", "a swarm of fireflies at dusk",
    # mythical / fantasy
    "a coiled emerald dragon", "a phoenix rising from embers", "a moonlit kelpie",
    "a stone golem awakening", "a griffin on a spire", "a sea serpent surfacing",
    "a forest spirit of moss and bark", "a wizard in a candlelit tower", "an elven archer in a glade",
    "a giant turtle carrying a forest", "a sleeping titan overgrown with vines",
    # objects / still life
    "an ornate pocket watch", "a cracked porcelain mask", "a glowing alchemical flask",
    "a ship in a bottle", "an antique brass telescope", "a spread of tarot cards",
    "a music box mid-melody", "a ring of ancient keys", "a still life of split pomegranates",
    "a bowl of steaming ramen", "a tiered patisserie cake", "a market stall of spices",
    "a vintage typewriter by a window", "a single rose in a crystal vase",
    # landscapes
    "a terraced rice valley at sunrise", "a frozen waterfall", "a field of lavender in wind",
    "a desert of red dunes", "a misty fjord", "a volcanic black-sand beach",
    "a canyon river bend", "an alpine meadow after rain", "a bamboo forest in fog",
    "rolling Tuscan hills with cypress", "a glacial lagoon with icebergs", "salt flats mirroring the sky",
    "a cherry-blossom avenue", "a storm rolling over wheat fields",
    # architecture / interiors
    "a half-ruined cathedral", "a floating temple in the clouds", "a brutalist concrete library",
    "a winding medina street", "a glass skyscraper at twilight", "a lighthouse on a cliff",
    "an overgrown abandoned station", "a spiral stone staircase", "a grand library reading room",
    "a Gothic cathedral interior", "a cozy cabin with a fireplace", "a Japanese tea house",
    "a Moroccan riad courtyard", "a sunlit artist's studio", "a greenhouse full of ferns",
    # vehicles
    "a vintage steam locomotive", "a wooden sailing galleon", "a chrome hot rod",
    "a weathered fishing trawler", "a hot-air balloon over hills", "a derelict submarine",
    "a biplane over patchwork fields", "a cable car climbing a peak",
    # plants / botanical / underwater / space
    "a single luminous mushroom", "a tangle of bioluminescent vines", "a bonsai in bloom",
    "a coral reef teeming with fish", "a diver among jellyfish", "a sunken statue garden",
    "a nebula cradling newborn stars", "a ringed gas giant from its moon", "an eclipse over a barren world",
    "the aurora over a snowy forest", "a comet streaking past mountains",
    # sci-fi
    "a derelict orbital ringworld", "a robot tending a garden", "a cybernetic street vendor",
    "a colony dome on a red planet", "a swarm of repair drones", "a neon-soaked rain-slick alley",
    "a cargo freighter near a gas giant", "an android contemplating the sea",
    # scenes / moments
    "a foggy harbor at dawn", "a rain-streaked cafe window", "a campfire under the stars",
    "an empty cinema aglow", "a snowy village at dusk", "a lone hiker above the clouds",
    "children flying kites on a hill", "a couple under one umbrella",
]

MEDIUMS = [
    "oil painting", "watercolor", "gouache", "3D render", "Octane render", "photograph",
    "cinematic film still", "concept art", "digital illustration", "ink drawing",
    "charcoal sketch", "pixel art", "paper collage", "risograph print", "matte painting",
    "pencil study", "acrylic on canvas", "linocut print", "pastel drawing", "vector illustration",
]
AESTHETICS = [
    "impressionist", "surrealist", "baroque", "art nouveau", "brutalist", "vaporwave",
    "ukiyo-e", "art deco", "minimalist", "gothic", "cyberpunk", "romanticist", "bauhaus",
    "folk-art", "renaissance", "pre-Raphaelite", "fauvist", "constructivist", "studio-ghibli inspired",
    "low-poly", "cel-shaded",
]
LIGHTING = [
    "golden-hour light", "soft studio softbox lighting", "dramatic rim lighting",
    "volumetric god rays", "neon glow", "warm candlelight", "flat overcast light",
    "moody chiaroscuro", "cool moonlight", "harsh midday sun", "backlit silhouette",
    "bioluminescent glow", "dappled forest light", "blue-hour twilight", "soft window light",
    "Rembrandt lighting",
]
MOODS = [
    "serene", "foreboding", "melancholic", "whimsical", "triumphant", "dreamlike",
    "tense", "nostalgic", "ecstatic", "solemn", "playful", "eerie", "tranquil", "epic",
]
COMPOSITIONS = [
    "extreme close-up", "wide establishing shot", "low-angle hero shot", "bird's-eye view",
    "symmetrical centered framing", "rule-of-thirds composition", "Dutch-angle framing",
    "shallow depth of field", "long telephoto compression", "fisheye perspective",
    "macro detail", "leading lines",
]
PALETTES = [
    "a muted earth-tone palette", "vivid complementary colors", "a cold monochrome blue palette",
    "warm amber and teal tones", "pastel candy colors", "high-contrast black and white",
    "iridescent pearlescent hues", "a limited two-color palette", "rich jewel tones",
    "washed-out faded colors", "warm autumnal hues", "cool desaturated greys",
]
QUALITY = [
    "highly detailed", "intricate textures", "sharp focus", "8k", "fine grain", "crisp linework",
    "painterly brushwork", "soft bokeh", "photorealistic", "atmospheric",
]


def build(rng: random.Random) -> str:
    parts = [f"{rng.choice(SUBJECTS)}, {rng.choice(MEDIUMS)}"]
    if rng.random() < 0.8:
        parts.append(f"in a {rng.choice(AESTHETICS)} style")
    if rng.random() < 0.85:
        parts.append(rng.choice(LIGHTING))
    if rng.random() < 0.6:
        parts.append(rng.choice(COMPOSITIONS))
    if rng.random() < 0.6:
        parts.append(f"{rng.choice(MOODS)} mood")
    if rng.random() < 0.5:
        parts.append(rng.choice(PALETTES))
    if rng.random() < 0.7:
        parts.extend(rng.sample(QUALITY, k=rng.randint(1, 2)))
    return ", ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--seed-file", default=os.path.join(here, "prompts_good100.txt"),
                    help="curated prompts to include verbatim (the quality bar)")
    ap.add_argument("--out", default=os.path.join(here, "prompts_1000.txt"))
    args = ap.parse_args(argv)

    out, seen = [], set()
    # 1) seed with the curated, hand-written prompts (skip comments/blanks)
    if os.path.exists(args.seed_file):
        for line in open(args.seed_file):
            s = line.strip()
            if s and not s.startswith("#") and s not in seen:
                seen.add(s); out.append(s)
    n_curated = len(out)

    # 2) fill the rest with generated diverse good prompts
    rng = random.Random(args.seed)
    attempts = 0
    while len(out) < args.n and attempts < args.n * 300:
        attempts += 1
        p = build(rng)
        if p not in seen:
            seen.add(p); out.append(p)
    if len(out) < args.n:
        raise RuntimeError(f"only {len(out)} unique; enlarge vocab or lower --n")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[gen_prompts_good] wrote {len(out)} prompts "
          f"({n_curated} curated + {len(out)-n_curated} generated) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
