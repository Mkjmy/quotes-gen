# SHITTY QUOTE GENERATOR v2.8 (The "Deep" Update)

This thing tries to be deep but it's still just a bunch of scripts glued together. Now it uses Trigrams to sound like it actually has a brain, knows about "Themes", and can export pretty pictures so you can pretend to be a philosopher on the internet.

## Setup (Do this or don't expect images)
The machine needs its own space (venv) to handle pictures without breaking your system.
```bash
python3 -m venv venv
./venv/bin/python -m pip install Pillow
```

## How to use (if you can read)

### 1. Make some "wisdom"
```bash
# Basic random nonsense
python main.py generate --num_quotes 5

# Act like you're from a specific theme
python main.py generate --theme general

# Make a whole block of "wisdom" (a paragraph)
python main.py generate --paragraph

# Make a long paragraph (15 sentences) and turn it into a PNG
python main.py generate --paragraph --sentences 15 --image
```

### 2. Make the brain bigger
Run this after you rate stuff (+/-) or dump new `.txt` files into `data/my_quotes/`:
```bash
python main.py learn
```

### 3. The "I want all the images" button
Convert your entire history from the CSV into a folder full of purple cloud images:
```bash
python main.py export
```

## Stuff you should know
- **Themes:** Go to `data/my_quotes/`, make a folder (e.g., `sad_boiz`), dump `.txt` files there, and run `learn`.
- **Background:** It uses `clound.jpg`. If you hate purple clouds, replace it with your own file but keep the same name.
- **Output:** Everything pretty goes into `output_images/`. 
- **Tracing:** Every image has a tiny "Seed" (ID) at the bottom. Use it to find out which specific thought created that masterpiece.

## Important Note
It still has a 20% "Chaos Factor". If it says something completely stupid, that's art. Don't blame me.

Have fun or whatever.
