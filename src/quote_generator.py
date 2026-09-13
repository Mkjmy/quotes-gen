import random
import argparse
import os
import json
import re
import base64
import hashlib
import sys
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

import pos_tagger as pt

# ---------------------------------------------------------------------------
# Corpus & n-gram config
# ---------------------------------------------------------------------------
_DEFAULT_CORPUS = [
    "data/my_quotes/general/fortune_corpus.txt",
    "data/my_quotes/general/real_quotes.txt",
]
_MY_QUOTES_ROOT = "data/my_quotes"
_MARKOV = {}

_SEED_MAP_PATH = "models/seed_map.json"
_END_PUNCT = {".", "!", "?"}
_BAD_START = {".", ",", ";", "!", "?", "'s", "'", "qotd", "qotd:",
              "of", "by", "in", "on", "at", "with", "from", "for", "under",
              "through", "beyond", "between", "within", "into", "without"}
_JUNK_TOKENS = {
    "knghtbrd", "swanson", "smirnoff", "hardcore", "paradoxum",
    "dells", "hanukkah", "pcline", "gosper", "linus", "bankhead",
}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def _corpus_files_for(theme):
    """Prefer every *.txt under data/my_quotes/<theme>/, else the default corpus."""
    theme_dir = os.path.join(_MY_QUOTES_ROOT, theme)
    if os.path.isdir(theme_dir):
        files = sorted(os.path.join(theme_dir, f) for f in os.listdir(theme_dir) if f.endswith(".txt"))
        if files:
            return files
    return _DEFAULT_CORPUS


def _clean_corpus_sentences(theme="general"):
    verbs = {"is", "are", "was", "were", "can", "will", "must", "should",
             "would", "could", "may", "might", "don't", "won't", "have",
             "has", "had", "be", "been", "do", "does", "did", "become",
             "becomes", "need"}
    sents = []
    for path in _corpus_files_for(theme):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8", errors="ignore"):
            orig = line.strip()
            if not orig:
                continue
            if any(ch.isdigit() for ch in orig):
                continue
            if any(not (ch.isalpha() or ch.isspace() or ch in ".,!?;:'\"" ) for ch in orig):
                continue
            if any(re.match(r"[A-Z][a-z]{2,}", w) for w in orig.split()[1:]):
                continue
            s = orig.lower()
            toks = re.findall(r"[a-z']+|[.,!?;]", s)
            if len(toks) < 5 or len(toks) > 16:
                continue
            if toks[-1] not in _END_PUNCT:
                continue
            if any(len(w) > 11 for w in toks):
                continue
            if toks[0] in _BAD_START:
                continue
            if any(w in _JUNK_TOKENS for w in toks):
                continue
            if not any(w in verbs or pt.assign_role_by_pattern(w) in ("VERB", "VING", "MODAL") for w in toks):
                continue
            sents.append(toks)
    return sents


def _load_markov(theme="general"):
    if theme in _MARKOV:
        return _MARKOV[theme]
    sents = _clean_corpus_sentences(theme)
    starts = Counter()
    f1 = defaultdict(Counter)
    f2 = defaultdict(Counter)
    for s in sents:
        if s[0] in _BAD_START:
            continue
        starts[s[0]] += 1
        for i in range(len(s) - 1):
            f1[s[i]][s[i + 1]] += 1
        for i in range(len(s) - 2):
            f2[(s[i], s[i + 1])][s[i + 2]] += 1
    markov = {"sents": sents, "starts": starts, "f1": dict(f1), "f2": dict(f2)}
    _MARKOV[theme] = markov
    return markov


# ---------------------------------------------------------------------------
# Token-level sampling: each next word comes from the previous tokens only.
# Chaos is reinterpreted as sampling temperature (no word substitution).
# ---------------------------------------------------------------------------
def _sample(keys, weights, temperature):
    if temperature <= 0.05:
        total = sum(weights)
        best_idx = max(range(len(keys)), key=lambda i: weights[i])
        if weights[best_idx] / total >= 0.5:
            return keys[best_idx]
    scaled = [max(w, 1e-9) ** (1.0 / temperature) for w in weights]
    return random.choices(keys, weights=scaled, k=1)[0]


def _wc(counter, temperature=1.0):
    keys = list(counter.keys())
    if not keys:
        return None
    return _sample(keys, [counter[k] for k in keys], temperature)


def _next_token(m, tokens, temperature):
    """Blend trigram (70%) + bigram (30%) continuation from the previous tokens."""
    pairs = {}
    if len(tokens) >= 2:
        trig = m["f2"].get((tokens[-2], tokens[-1]))
        if trig:
            for tok, cnt in trig.items():
                pairs[tok] = pairs.get(tok, 0.0) + 0.7 * cnt
    bi = m["f1"].get(tokens[-1])
    if bi:
        for tok, cnt in bi.items():
            pairs[tok] = pairs.get(tok, 0.0) + 0.3 * cnt
    if not pairs:
        return None
    items = list(pairs.items())
    return _sample([t for t, _ in items], [w for _, w in items], temperature)


def build_sentence(theme="general", chaos=0.6):
    m = _load_markov(theme)
    temperature = 0.4 + chaos * 1.0
    for _ in range(30):
        w1 = _wc(m["starts"], temperature)
        if w1 is None or w1 in _BAD_START:
            continue
        toks = [w1]
        guard = 0
        while len(toks) < 20 and guard < 30:
            guard += 1
            nxt = _next_token(m, toks, temperature)
            if nxt is None:
                break
            if nxt in {",", ";", "!"}:
                if len(toks) >= 6:
                    break
                continue
            toks.append(nxt)
            if nxt in _END_PUNCT:
                break
        if len(toks) >= 5 and toks[-1] in _END_PUNCT:
            return toks
    if m["sents"]:
        return m["sents"][random.randrange(len(m["sents"]))]
    return ["everything means something."]


# ---------------------------------------------------------------------------
# Vowel-sound helpers for a/an
# ---------------------------------------------------------------------------
_SILENT_H = {"hour","hours","honest","honesty","honor","honour","honors","honours",
              "honorable","heir","heirs","heiress","herb","herbs","herbal","homage"}
_U_LONG = {"unit","units","unique","user","users","united","universal","universities",
           "university","uniform","uniforms","useful","usual","usually","union","unions",
           "utah","eunuch","euphemism","eulogy"}

def _is_vowel_sound(word):
    w = word.lower()
    if w in _SILENT_H:
        return True
    if w[0] in "aeiou" and w not in _U_LONG:
        return True
    return False


# ---------------------------------------------------------------------------
# Seeds (tracing)
# ---------------------------------------------------------------------------
def quote_to_seed(text):
    """Map a quote to a fixed-length seed (always 67 chars).

    Reversible only through the seed registry (models/seed_map.json), not by
    the seed itself. Hash is deterministic, so the same quote always yields the
    same 67-char seed on any machine.
    """
    seed = _seed_code(text)
    registry = _load_seed_registry()
    if seed not in registry:
        registry[seed] = text.strip()
        _save_seed_registry(registry)
    return seed


def seed_to_quote(seed):
    """Recover the quote behind a 67-char seed, or None if unknown."""
    if not seed:
        return None
    seed = seed.strip()
    registry = _load_seed_registry()
    q = registry.get(seed)
    if q:
        return q
    try:
        q = base64.urlsafe_b64decode(seed.encode("ascii")).decode("utf-8").strip()
        if len(q) >= 10 and q[-1] in ".!?":
            return q
    except Exception:
        pass
    return None


def _seed_code(text):
    """50-byte blake2b -> urlsafe base64, unpadded -> exactly 67 chars."""
    digest = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=50).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _load_seed_registry():
    path = _SEED_MAP_PATH
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_seed_registry(registry):
    try:
        os.makedirs(os.path.dirname(_SEED_MAP_PATH), exist_ok=True)
        with open(_SEED_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def fix_articles(text):
    words = text.split()
    out = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if w in ("a", "an") and nxt and nxt[0].isalpha():
            vs = _is_vowel_sound(nxt)
            if w == "a" and vs:
                out.append("an")
            elif w == "an" and not vs:
                out.append("a")
            else:
                out.append(w)
        else:
            out.append(w)
    return " ".join(out)


def fix_case(text):
    text = text.strip()
    words = text.split()
    words = ["I" if w == "i" else w for w in words]
    if words:
        words[0] = words[0].capitalize()
    text = " ".join(words)
    if text and text[-1] not in ".?!":
        text += "."
    text = text.replace("  ", " ").replace(" ,", ",").replace(" ?", "?").replace(" .", ".").replace(" ;", ";")
    return text


def sanity_check(text):
    words = [w for w in text.replace(",", " ").replace(";", " ").replace("—", " ").split()]
    if len(words) < 4:
        return False
    if max(len(w) for w in words) > 15:
        return False
    if sum(1 for w in words if len(w) > 10) > 1:
        return False
    verbs = {"is", "are", "was", "were", "can", "will", "must", "should", "could", "may", "might",
             "becomes", "seems", "do", "have", "had", "has", "don't", "never", "you", "what"}
    if not any(w in verbs or (w.isalpha() and pt.assign_role_by_pattern(w) in ("VERB", "VING", "MODAL")) for w in words):
        return False
    return True


# ---------------------------------------------------------------------------
# Generation entry points
# ---------------------------------------------------------------------------
def generate_phrase(theme="general", chaos=0.6):
    for _ in range(25):
        toks = build_sentence(theme, chaos)
        sentence = fix_articles(" ".join(toks))
        sentence = fix_case(sentence)
        if sanity_check(sentence):
            return sentence
    m = _load_markov(theme)
    if not m["sents"]:
        return "Everything has its own silence."
    return fix_case(fix_articles(" ".join(random.choice(m["sents"]))))


def generate_full_quote(theme="general", chaos=0.6):
    return generate_phrase(theme, chaos)


def generate_paragraph(theme="general", num_sentences=8, chaos=0.6):
    transitions = ["Moreover, ", "Thus, ", "Yet, ", "In the end, ", "Therefore, ",
                   "Consequently, ", "Simply put, ", "Indeed, ", "Beyond this, "]
    paragraph = []
    for i in range(num_sentences):
        sentence = generate_full_quote(theme=theme, chaos=chaos)
        if i > 0 and random.random() < 0.5:
            trans = random.choice(transitions)
            sentence = trans + sentence[0].lower() + sentence[1:]
        paragraph.append(sentence)
    return " ".join(paragraph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_quotes", type=int, default=5)
    parser.add_argument("--theme", default="general")
    parser.add_argument("--chaos", type=float, default=0.6, help="0..1 sampling temperature (0 = closest to corpus)")
    parser.add_argument("--paragraph", action="store_true")
    parser.add_argument("--sentences", type=int, default=8, help="Number of sentences in the paragraph")
    parser.add_argument("--raw", action="store_true", help="Print only the quote text")
    parser.add_argument("--svg", action="store_true", help="Export to a stylish SVG image")
    parser.add_argument("--image", action="store_true", help="Export to a professional PNG image (requires venv)")
    parser.add_argument("--seed", help="Reproduce a specific quote from its seed (use --to-seed to make one)")
    parser.add_argument("--to-seed", help="Print the seed for a given quote, then exit")
    parser.add_argument("--quote", help="Pump an outside quote straight through the pipeline (seed + optional svg/image)")
    args = parser.parse_args()

    if args.to_seed:
        print(quote_to_seed(args.to_seed))
        sys.exit(0)

    chaos = max(0.0, min(1.0, args.chaos))
    theme = args.theme if os.path.isdir(os.path.join("data", "my_quotes", args.theme)) else "general"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.paragraph:
        final_text = generate_paragraph(theme=theme, num_sentences=args.sentences, chaos=chaos)
        if args.raw:
            print(final_text)
        else:
            print(f"\n--- [THEME: {theme.upper()}] DEEP PERSPECTIVE ---\n")
            print(final_text)
            print("\n" + "=" * 60 + "\n")
        base_name = f"paragraph_{theme.upper()}_{timestamp}"
        if args.svg:
            from svg_generator import generate_svg
            generate_svg(final_text, theme=theme.upper(), output_path=f"{base_name}.svg")
        if args.image:
            output_dir = "output_images"
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            out_path = os.path.join(output_dir, f"{base_name}.png")
            venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
            if os.path.exists(venv_python):
                subprocess.run([venv_python, "src/image_generator.py", final_text, theme.upper(), out_path, "Paragraph"])
    else:
        output_dir = "output_images"
        if args.image and not os.path.exists(output_dir): os.makedirs(output_dir)
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
        import uuid

        seeded_quotes = []
        if args.quote:
            seeded_quotes = [args.quote]
        elif args.seed:
            q = seed_to_quote(args.seed)
            if q is not None:
                seeded_quotes = [q]

        for i in range(len(seeded_quotes) or args.num_quotes):
            final_text = seeded_quotes[i] if seeded_quotes else generate_full_quote(theme=theme, chaos=chaos)
            q_id = str(uuid.uuid4())
            print(f"[{theme.upper()}|chaos{chaos:.2f}] {final_text}" if not args.raw else final_text)

            base_name = f"quote_{theme.upper()}_{timestamp}_{i+1}"
            if args.svg:
                from svg_generator import generate_svg
                generate_svg(final_text, theme=theme.upper(), output_path=f"{base_name}.svg")
            if args.image:
                out_path = os.path.join(output_dir, f"{base_name}.png")
                if os.path.exists(venv_python):
                    subprocess.run([venv_python, "src/image_generator.py", final_text, theme.upper(), out_path, q_id])