# SHITTY QUOTE GENERATOR v1.0

This thing tries to be deep but it's just a bunch of scripts glued together. It makes quotes. Sometimes they're good, mostly they're just weird.

## What's it do?
- **Make quotes:** It mashes words together and hopes for the best.
- **Learn shit:** If you tell it a quote is good or trash, it remembers.
- **Steal style:** You can feed it real quotes from famous people and it'll try to act smart.

## How to use (if you can read)

### 1. Make some "wisdom"
```bash
python main.py generate --num_quotes 5
```
Add `--rate` if you want to judge the machine's output.

### 2. Make the brain bigger
Run this after you rate stuff or add new data:
```bash
python main.py learn
```

### 3. Feed it the good stuff
Go to `data/my_quotes/`. Dump any `.txt` file in there with real quotes (one per line). 
**IMPORTANT:** Don't include the author's name (like ~Einstein), the script is too dumb to know who that is. Just the words. 
Then run `python main.py learn` again.

## Stuff you should know
- It's buggy. Deal with it.
- If you feed it garbage, it will spit out garbage. 
- The "AI" here is basically just a glorified calculator.

Have fun or whatever.
