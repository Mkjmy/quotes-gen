import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_words_from_file(filename):
    """Loads words from a text file, one word per line."""
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'r') as f:
        return {word.strip().lower() for word in f if word.strip()}

# These print statements are useful for debugging during development,
# but might be removed in a production pos_tagger.
common_words_raw = load_words_from_file('commonwords.txt')

all_words_raw = load_words_from_file('words.txt')

all_words = all_words_raw.union(common_words_raw)


_FUNC_CANDIDATES = {
    'a', 'an', 'the', 'to', 'in', 'on', 'of', 'and', 'but', 'or', 'if', 'then', 'than', 'as',
    'about', 'with', 'by', 'at', 'from', 'into', 'over', 'under', 'through', 'before', 'after',
    'while', 'when', 'where', 'how', 'why', 'what', 'which', 'who', 'whom', 'whose',
    'wherever', 'whenever', 'however', 'because', 'since', 'until', 'unless', 'though',
    'although', 'even', 'only', 'also', 'just', 'too', 'very', 'quite', 'rather', 'still',
    'yet', 'already', 'soon', 'always', 'never', 'often', 'sometimes', 'usually', 'rarely',
    'seldom', 'ever', 'almost', 'nearly', 'enough', 'much', 'many', 'more', 'most', 'less',
    'least', 'few', 'little', 'another', 'other', 'both', 'all', 'several', 'some', 'any', 'no',
    'every', 'each', 'either', 'neither', 'one', 'two', 'three', # Numbers can act functionally
    'not', # Crucial functional word
    'so', 'such', # Conjunction-like/determiner-like
    'up', 'down', 'out', 'off', 'back', 'forward', 'away', # Adverbial particles
    'again', 'ago'
}
FUNC = {w for w in _FUNC_CANDIDATES if w in all_words}

_PRON_CANDIDATES = {
    'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'they', 'them',
    'mine', 'yours', 'his', 'hers', 'ours', 'theirs',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
    'everyone', 'everybody', 'someone', 'somebody', 'anyone', 'anybody',
    'noone', 'nobody', 'something', 'anything', 'nothing', 'everything',
    'who', 'whom', 'whose', 'which', 'what',
    'this', 'that', 'these', 'those'
}
PRON = {w for w in _PRON_CANDIDATES if w in all_words}


_MODAL_CANDIDATES = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
MODAL = {w for w in _MODAL_CANDIDATES if w in all_words}


_DET_CANDIDATES = {
    'a', 'an', 'the',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'this', 'that', 'these', 'those',
    'some', 'any', 'no', 'every', 'each', 'either', 'neither', 'both', 'all',
    'few', 'many', 'most', 'several', 'enough', 'more', 'less', 'little', 'much',
    'another', 'other', 'one'
}
DET = {w for w in _DET_CANDIDATES if w in all_words}

def assign_role_by_pattern(word):
    word_lower = word.lower()

    if word_lower in FUNC:
        return 'FUNC'
    if word_lower in PRON:
        return 'PRON'
    if word_lower in MODAL:
        return 'MODAL'
    if word_lower in DET:
        return 'DET'

    if word_lower.endswith("ing"):
        return 'VING'
    if word_lower.endswith("ly"):
        return 'ADV'
    if word_lower.endswith(("able", "ous", "ful", "less", "ive", "ic", "al")):
        return 'ADJ'
    if word_lower.endswith(("ed", "ize", "ify", "ate")):
        return 'VERB'

    return 'NOUN'