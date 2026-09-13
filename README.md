# SHITTY QUOTE GENERATOR v3

You throw text at it, it builds a Markov chain, then it spits out "philosophical" sentences where every word has genuinely existed somewhere in the text you fed it. No sentence templates, no word-swapping — the previous token decides the next token.

In plain terms: it never makes anything up. It only says what the source already said, just scrambled in an order that makes you look like a philosopher.

## Setup

The machine needs its own space (venv) to draw pictures without breaking your system:

```bash
python3 -m venv venv
./venv/bin/python -m pip install Pillow
```

## Running

```bash
# Basic nonsense
python main.py generate --num_quotes 5

# A whole block of "deep" (8 sentences)
python main.py generate --paragraph

# 15 sentences + export a PNG
python main.py generate --paragraph --sentences 15 --image

# Pump your OWN sentence through the pipeline (seed included, image if wanted)
python main.py generate --quote "Don't make a lethal teaspoon." --image
```

Real output, nothing hidden:

```
[GENERAL|chaos0.60] Don't make a lethal teaspoon.
```

### Flags

- `--num_quotes N` — how many quotes to print
- `--theme X` — use the corpus folder `data/my_quotes/X/`
- `--chaos 0..1` — the **temperature** of the token-selection step. 0 = loyal to the source, 1 = semantically airborne (still only picks words from the corpus)
- `--paragraph`, `--sentences N` — generate a whole paragraph / number of sentences
- `--raw` — print only the text, no seasoning
- `--svg` — export a gradient SVG file
- `--image` — export a PNG to `output_images/`
- `--seed X` — rebuild a quote from its seed
- `--to-seed "quote"` — turn a quote into a seed, then walk away

## Corpus & "themes"

Each folder in `data/my_quotes/` is a theme. Drop `.txt` files in, one quote per line, done. The generator reads them directly — no `learn` needed.

```
data/my_quotes/
├── general/
│   ├── fortune_corpus.txt   # ~15k handled "prophecies"
│   └── real_quotes.txt      # 99 allegedly real quotes
└── sad_boiz/
    └── depression.txt       # your contribution
```

No theme folder → the generator falls back to `general`. The thicker the corpus, the further the Markov chain walks before dying mid-sentence — the longer your quotes.

## Images

- Background is `clound.jpg` (yes, typo, it's art). It's actually almost black all over — a night sky. Want color? Swap the file but keep the name.
- Text uses **JetBrains Mono ExtraLight / Thin**, resolved through fontconfig; without those it falls back to a sad default font.
- Text is drawn on its own layer and **screen-blended** onto the scene — a faded Photoshop-overlay look. Intentionally subtle.
- Every image has a tiny seed at the bottom so you know which sentence birthed it.

## `learn` and `export`

```bash
python main.py learn     # still runs. writes a file nobody reads anymore.
python main.py export    # turns the whole history CSV into an armada of images.
```

`models/learned_parameters.json` exists, looks fancy, but the generator broke up with it a while ago. Don't feel bad for it.

## Misc

- `--raw` + `--num_quotes 30` and read for fun, guaranteed at least one keeper.
- The error "everything means something." means the corpus is empty — you broke it.
- ImageMagick inside `src/image_pro.py` still sits there like a historical landmark. Don't touch it.
- Everything pretty lands in `output_images/`.

Have fun or whatever.