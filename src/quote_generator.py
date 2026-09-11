import random
import argparse
import os
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

import pos_tagger as pt

LEARNED_DATA = {"themes": {}}
CURRENT_THEME = "general"

# ---------------------------------------------------------------------------
# Word pools
# ---------------------------------------------------------------------------
NORMAL_NOUNS = [
    "life", "way", "time", "thing", "heart", "mind", "world", "people", "work",
    "morning", "evening", "rain", "door", "house", "floor", "window", "sky",
    "bread", "bucket", "garden", "pocket", "letter", "engine", "traffic",
    "staircase", "wardrobe", "bed", "kitchen", "river", "forest", "mountain",
]
NORMAL_VERBS = [
    "build", "break", "rise", "fall", "speak", "listen", "seek", "keep",
    "polish", "return", "scatter", "soften", "sharpen", "harvest", "carry",
    "bend", "move", "stay", "loosen", "plant", "open", "close", "shape",
    "find", "follow", "hold", "harden", "gather", "surrender",
]
NORMAL_ADJS = [
    "quiet", "broken", "hollow", "golden", "gentle", "patient", "generous",
    "empty", "full", "tender", "sharp", "soft", "old", "new", "warm", "tired",
]
NORMAL_ADVS = ["softly", "deeply", "slowly", "loudly", "quietly", "eventually", "daily"]
NORMAL_VINGS = ["dreaming", "breaking", "rising", "fading", "burning", "growing", "waiting"]

DEEP_NOUNS = [
    "silence", "courage", "truth", "shadow", "echo", "horizon", "stillness",
    "emptiness", "absence", "gravity", "patience", "salt", "hummingbird",
    "whetstone", "threshold", "archipelago", "copper", "rumor", "bloom",
    "mud", "cold", "dawn", "wound", "mercy", "discipline", "ocean", "cedar",
]
DEEP_VERBS = [
    "surrender", "unlearn", "hollow", "startle", "persist", "ripen", "smolder",
    "linger", "stagger", "inhabit", "deny", "forgive", "endure", "withdraw",
    "dissolve", "whisper",
]
DEEP_ADJS = [
    "tender", "ruthless", "perennial", "buried", "relentless", "vigilant",
    "untended", "intact", "faint", "unfair", "patient", "salt-cured",
    "sleepless", "perpetual",
]
DEEP_ADVS = ["quietly", "somewhere", "utterly", "patiently", "slowly", "carefully"]
DEEP_VINGS = ["dreaming", "unlearning", "fermenting", "crouching", "sinking", "burning"]

ABSURD_NOUNS = [
    "goose", "sprocket", "microwave", "hamster", "puddle", "toaster",
    "eyebrow", "knee", "teaspoon", "ottoman", "spatula", "pickle",
    "marmalade", "fog", "zebra", "lawnmower", "accordion", "turnstile",
    "mezzanine", "credenza", "couscous", "paperweight", "whisker",
    "pastrami", "trombone", "subfloor", "knapsack", "hydrangea", "lint",
    "doorknob", "javelin", "stapler", "binder", "cabbage", "firewagon",
    "gerbil", "toenail", "hedgerow", "vending machine", "dustbuster",
]
ABSURD_VERBS = [
    "laminate", "marinate", "hitch", "oil", "fold", "shuffle", "staple",
    "thaw", "park", "wiggle", "rehydrate", "straddle", "shave", "caulk",
    "recalibrate", "baste",
]
ABSURD_ADJS = [
    "caffeinated", "damp", "reheated", "folded", "squeaky", "preloved",
    "bi-folding", "moth-eaten", "upside-down", "semi-permanent", "teriyaki",
    "fuzz-covered",
]
ABSURD_ADVS = ["sideways", "biweekly", "windward", "repeatedly", "northwards"]
ABSURD_VINGS = ["fermenting", "wobbling", "marinating", "reassembling", "distempering", "unscrewing"]

POOLS = {
    "NOUN": {"normal": NORMAL_NOUNS, "deep": DEEP_NOUNS, "absurd": ABSURD_NOUNS},
    "VERB": {"normal": NORMAL_VERBS, "deep": DEEP_VERBS, "absurd": ABSURD_VERBS},
    "ADJ": {"normal": NORMAL_ADJS, "deep": DEEP_ADJS, "absurd": ABSURD_ADJS},
    "ADV": {"normal": NORMAL_ADVS, "deep": DEEP_ADVS, "absurd": ABSURD_ADVS},
    "VING": {"normal": NORMAL_VINGS, "deep": DEEP_VINGS, "absurd": ABSURD_VINGS},
}

AUTHOR_ADJS = DEEP_ADJS + ["silent", "tired", "foreign", "caffeinated", "permanent", "indifferent"]
AUTHOR_NOUNS = [
    "goose", "blacksmith", "accountant", "fog", "veteran", "architect",
    "janitor", "shepherd", "llama", "oracle", "technician", "gnome", "corvid",
]

PAST_TENSE = {
    "was", "were", "had", "got", "became", "been", "did", "made", "went",
    "took", "said", "saw", "came", "found", "left", "felt", "ran", "spoke",
    "wrote", "kept", "built", "broke", "held", "sat", "stood",
    "lost", "gave", "knew", "began", "fell", "grew", "ate", "drew", "drank",
    "beat", "drove", "forgot", "hid", "rose", "woke", "sent", "spent",
    "brought", "caught", "bought", "taught", "sought", "threw", "wore",
    "chose", "froze", "bore", "tore", "told", "heard", "met", "won", "hit",
}

EXCLUDE_WORDS = {
    "mr", "mrs", "ms", "dr", "oh", "uh", "hmm", "yeah", "yep", "nope", "ooh",
    "wow", "hey", "hello", "wait", "please", "really", "far", "via", "etc",
    "vs", "str", "char", "foo", "bar", "sin", "cos", "tan", "int", "obj",
    "null", "true", "false", "ll", "ve", "re", "don", "won", "im", "s",
    "st", "nd", "rd", "th", "e", "o", "u", "whether", "probably", "quite",
    "almost", "maybe", "though", "if", "however", "therefore",
}

# ---------------------------------------------------------------------------
# Connector / grammar vocabulary
# ---------------------------------------------------------------------------
ARTICLES = ["the", "a", "your", "my", "our", "this", "that", "every", "each", "one", "no", "some"]
ART_SOFT = ["the", "a", "your", "this", "every", "one"]
PREPS = ["of", "in", "on", "from", "with", "through", "beyond", "between",
         "within", "against", "toward", "under", "into", "without", "across", "despite"]
CONJS = ["and", "but", "yet", "so", "because", "while", "when", "though", "until", "still"]
MODALS = ["can", "will", "must", "may", "might", "should", "could"]
COP = ["is", "are"]
SUBJS = ["you", "we", "one", "i"]
SUBJ2 = ["you", "we", "one"]
ALONE = ["for once", "at last", "at night", "in the end", "alone"]
END_FLAVOR = ["or so", "and yet", "this way", "perhaps"]
ADVF = ["only", "just", "simply", "never", "still", "always", "truly", "quietly",
        "slowly", "deeply", "finally", "patiently", "seldom", "sometimes"]

# ---------------------------------------------------------------------------
# Loading & pools
# ---------------------------------------------------------------------------
def load_learned_parameters(filepath="models/learned_parameters.json"):
    global LEARNED_DATA
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                LEARNED_DATA = json.load(f)
        except Exception as e:
            print(f"Error loading parameters: {e}")


def get_theme_dict():
    return LEARNED_DATA.get("themes", {}).get(CURRENT_THEME, {})


_BIG_POOLS_CACHE = "models/big_pools.json"
_big_pools = None


def build_theme_pools():
    """Tag the theme vocab and group content words by POS."""
    vocab = get_theme_dict().get("vocab", [])
    pools = {pos: [] for pos in POOLS}
    for w in vocab:
        if len(w) < 3 or len(w) > 9:
            continue
        if not w.isalpha() or w in EXCLUDE_WORDS:
            continue
        pos = pt.assign_role_by_pattern(w)
        if pos == "VERB" and (w.endswith("ed") or w in PAST_TENSE):
            continue
        if pos in pools:
            pools[pos].append(w)
    return pools


def load_big_pools():
    """POS pools built from data/words.txt (466k words), cached to JSON."""
    global _big_pools
    if _big_pools is not None:
        return _big_pools
    if os.path.exists(_BIG_POOLS_CACHE):
        try:
            with open(_BIG_POOLS_CACHE, encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded and loaded.get("words") == "words.txt":
                _big_pools = {pos: loaded.get(pos, []) for pos in POOLS}
                return _big_pools
        except Exception:
            pass
    pools = {pos: [] for pos in POOLS}
    try:
        for line in open("data/words.txt", encoding="utf-8", errors="ignore"):
            raw = line.strip()
            if not raw.islower() or not raw.isalpha():
                continue
            w = raw.lower()
            if len(w) < 3 or len(w) > 10 or w in EXCLUDE_WORDS:
                continue
            pos = pt.assign_role_by_pattern(w)
            if pos == "VERB" and (w.endswith("ed") or w in PAST_TENSE):
                continue
            if pos in pools:
                pools[pos].append(w)
    except Exception:
        pass
    _big_pools = pools
    try:
        os.makedirs("models", exist_ok=True)
        with open(_BIG_POOLS_CACHE, "w", encoding="utf-8") as f:
            json.dump({"words": "words.txt", **{pos: pools[pos] for pos in POOLS}}, f)
    except Exception:
        pass
    return pools


def pick_slot(pos, theme_pools, chaos):
    pools = POOLS[pos]
    theme = theme_pools.get(pos) or []
    normal = theme if theme else pools["normal"]
    r = random.random()
    if chaos > 0:
        cutoff_absurd = 0.20 + 0.55 * chaos
        cutoff_deep = cutoff_absurd + 0.25
        if r < cutoff_absurd:
            big = load_big_pools().get(pos) or []
            if random.random() < 0.65 and big:
                return random.choice(big)
            return random.choice(pools["absurd"])
        if r < cutoff_deep:
            big = load_big_pools().get(pos) or []
            if random.random() < 0.30 and big:
                return random.choice(big)
            return random.choice(pools["deep"])
    return random.choice(normal)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# N-gram engine — no templates at all.  Each next word is sampled from the
# words that actually followed the current word (or the previous two) in the
# learnt corpus.  Chaos only tweaks *which* words land in a slot: content
# words get swapped for absurd/theme words of the same part of speech, so the
# grammar comes straight from real quotes while the meaning goes sideways.
# ---------------------------------------------------------------------------
_MARKOV = None
_CORPUS_FILES = [
    "data/my_quotes/general/fortune_corpus.txt",
    "data/my_quotes/general/real_quotes.txt",
]
_END_PUNCT = {".", "!", "?"}
_BAD_START = {".", ",", ";", "!", "?", "'s", "'", "qotd", "qotd:",
              "of", "by", "in", "on", "at", "with", "from", "for", "under",
              "through", "beyond", "between", "within", "into", "without"}
_JUNK_TOKENS = {
    "knghtbrd", "swanson", "smirnoff", "hardcore", "paradoxum",
    "dells", "hanukkah", "pcline", "gosper", "linus", "bankhead",
}


def _clean_corpus_sentences():
    verbs = {"is", "are", "was", "were", "can", "will", "must", "should",
             "would", "could", "may", "might", "don't", "won't", "have",
             "has", "had", "be", "been", "do", "does", "did", "become",
             "becomes", "need"}
    sents = []
    for path in _CORPUS_FILES:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8", errors="ignore"):
            orig = line.strip()
            if not orig:
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


def _load_markov():
    global _MARKOV
    if _MARKOV is not None:
        return _MARKOV
    sents = _clean_corpus_sentences()
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
    _MARKOV = {"sents": sents, "starts": starts, "f1": dict(f1), "f2": dict(f2)}
    return _MARKOV


def _wc(counter):
    keys = list(counter.keys())
    if not keys:
        return None
    return random.choices(keys, weights=[counter[k] for k in keys], k=1)[0]


def _verb_agree(word, kind):
    if kind != "sg":
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh", "o")):
        return word + "es"
    return word + "s"


def _mayhem(tok, theme_pools, chaos):
    t = tok.lower()
    if t in _END_PUNCT or t in {",", ";", "'"} or len(t) < 3:
        return tok
    pos = pt.assign_role_by_pattern(t)
    if pos not in POOLS:
        return tok
    if random.random() >= 0.08 + 0.85 * chaos:
        return tok
    rep = pick_slot(pos, theme_pools, chaos)
    if pos == "VERB" and t.endswith("s") and not t.endswith(("is", "has", "was")):
        rep = _verb_agree(rep, "sg")
    return rep


def build_sentence(theme_pools, chaos):
    m = _load_markov()
    for _ in range(30):
        w1 = _wc(m["starts"])
        if w1 is None or w1 in _BAD_START:
            continue
        toks = [w1]
        guard = 0
        while len(toks) < 20 and guard < 30:
            guard += 1
            c = None
            if len(toks) >= 2:
                c = m["f2"].get((toks[-2], toks[-1]))
            if not c:
                c = m["f1"].get(toks[-1])
            if not c:
                break
            nxt = _wc(c)
            if nxt in {",", ";", "!"}:
                if len(toks) >= 6:
                    break
                continue
            nxt = _mayhem(nxt, theme_pools, chaos)
            toks.append(nxt)
            if nxt in _END_PUNCT:
                break
        if len(toks) >= 5 and toks[-1] in _END_PUNCT:
            return toks

    raw = random.choice(m["sents"])
    toks = []
    for t in raw[:20]:
        toks.append(_mayhem(t, theme_pools, chaos))
        if t in _END_PUNCT:
            break
    return toks


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


def maybe_glue(sentence, chaos):
    r = random.random()
    adjective = random.choice(AUTHOR_ADJS)
    noun = random.choice(AUTHOR_NOUNS)
    core = chaos > 0.15 and r < 0.10 + 0.15 * chaos
    if core:
        if sentence.endswith("."):
            return sentence[:-1] + f", said the {adjective} {noun}."
        if sentence.endswith("?"):
            return sentence[:-1] + f", asked the {adjective} {noun}."
        return sentence + f" -- the {adjective} {noun}"
    if chaos > 0.25 and r < 0.08:
        if sentence.endswith("."):
            flavor = random.choice(END_FLAVOR)
            return sentence[:-1] + f", {flavor}."
    return sentence


def generate_phrase(theme_pools, chaos=0.6):
    for _ in range(25):
        toks = build_sentence(theme_pools, chaos)
        sentence = fix_articles(" ".join(toks))
        sentence = fix_case(sentence)
        if sanity_check(sentence):
            return maybe_glue(sentence, chaos)
    m = _load_markov()
    toks = random.choice(m["sents"])
    sentence = fix_case(fix_articles(" ".join(toks)))
    return maybe_glue(sentence, 0.2)


def generate_full_quote(theme="general", chaos=0.6):
    global CURRENT_THEME
    CURRENT_THEME = theme
    theme_pools = build_theme_pools()
    return generate_phrase(theme_pools, chaos)


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
    parser.add_argument("--chaos", type=float, default=0.6, help="0..1 how unhinged the quotes get")
    parser.add_argument("--paragraph", action="store_true")
    parser.add_argument("--sentences", type=int, default=8, help="Number of sentences in the paragraph")
    parser.add_argument("--raw", action="store_true", help="Print only the quote text")
    parser.add_argument("--svg", action="store_true", help="Export to a stylish SVG image")
    parser.add_argument("--image", action="store_true", help="Export to a professional PNG image (requires venv)")
    args = parser.parse_args()

    load_learned_parameters()

    if not LEARNED_DATA.get("themes", {}):
        LEARNED_DATA["themes"] = {"general": {}}

    if args.theme not in LEARNED_DATA.get("themes", {}):
        args.theme = "general" if "general" in LEARNED_DATA.get("themes", {}) else list(LEARNED_DATA.get("themes", {}).keys())[0]

    chaos = max(0.0, min(1.0, args.chaos))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.paragraph:
        final_text = generate_paragraph(theme=args.theme, num_sentences=args.sentences, chaos=chaos)
        if args.raw:
            print(final_text)
        else:
            print(f"\n--- [THEME: {args.theme.upper()}] DEEP PERSPECTIVE ---\n")
            print(final_text)
            print("\n" + "=" * 60 + "\n")
        base_name = f"paragraph_{args.theme.upper()}_{timestamp}"
        if args.svg:
            from svg_generator import generate_svg
            generate_svg(final_text, theme=args.theme.upper(), output_path=f"{base_name}.svg")
        if args.image:
            output_dir = "output_images"
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            out_path = os.path.join(output_dir, f"{base_name}.png")
            venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
            if os.path.exists(venv_python):
                subprocess.run([venv_python, "src/image_generator.py", final_text, args.theme.upper(), out_path, "Paragraph"])
    else:
        output_dir = "output_images"
        if args.image and not os.path.exists(output_dir): os.makedirs(output_dir)
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
        import uuid

        for i in range(args.num_quotes):
            final_text = generate_full_quote(theme=args.theme, chaos=chaos)
            q_id = str(uuid.uuid4())
            print(f"[{args.theme.upper()}|chaos{chaos:.2f}] {final_text}" if not args.raw else final_text)

            base_name = f"quote_{args.theme.upper()}_{timestamp}_{i+1}"
            if args.svg:
                from svg_generator import generate_svg
                generate_svg(final_text, theme=args.theme.upper(), output_path=f"{base_name}.svg")
            if args.image:
                out_path = os.path.join(output_dir, f"{base_name}.png")
                if os.path.exists(venv_python):
                    subprocess.run([venv_python, "src/image_generator.py", final_text, args.theme.upper(), out_path, q_id])