import argparse
import csv
import json
import os
import math

from pos_tagger import all_words, assign_role_by_pattern, FUNC, PRON, MODAL, DET

# Keywords that give a 'philosophical' feel
PHILOSOPHICAL_KEYWORDS = {
    'time', 'soul', 'shadow', 'light', 'silence', 'echo', 'mirror', 'dream', 'truth',
    'void', 'path', 'star', 'mountain', 'river', 'wind', 'death', 'life', 'wisdom',
    'chaos', 'order', 'empty', 'full', 'nothing', 'everything', 'being', 'becoming',
    'nature', 'mind', 'heart', 'spirit', 'fear', 'courage', 'freedom', 'fate', 'destiny'
}

def get_pair_weight(index, total_tokens, raw_words_list):
    if total_tokens <= 1: return 0.0
    if index == 0 or index == total_tokens - 2: return 1.5 # Boost start/end
    return 0.5

def process_quote_list(quotes_data, min_confidence_count, ignore_threshold):
    pos_pair_raw_scores = {}
    pos_pair_confidence = {}
    pos_ending_raw_scores = {}
    pos_ending_confidence = {}
    word_trigram_raw_scores = {}
    word_trigram_confidence = {}
    
    # Store words that frequently appear in high-quality quotes
    theme_vocab = {}

    for quote_data in quotes_data:
        raw_words = quote_data['raw_words']
        sentence_score = quote_data['sentence_score']
        pos_sequence = ['COMMA' if w == ',' else assign_role_by_pattern(w) for w in raw_words]
        num_tokens = len(raw_words)

        for i, word in enumerate(raw_words):
            if word == ',': continue
            w_lower = word.lower()
            bonus = 2.0 if w_lower in PHILOSOPHICAL_KEYWORDS else 1.0
            theme_vocab[w_lower] = theme_vocab.get(w_lower, 0.0) + (sentence_score * bonus)

        # POS Bigrams
        for i in range(num_tokens - 1):
            p1, p2 = pos_sequence[i], pos_sequence[i+1]
            if p1 == 'COMMA' or p2 == 'COMMA': continue
            key = f"{p1},{p2}"
            theme_vocab[raw_words[i].lower()] = theme_vocab.get(raw_words[i].lower(), 0) + 0.1
            pos_pair_raw_scores[key] = pos_pair_raw_scores.get(key, 0.0) + sentence_score
            pos_pair_confidence[key] = pos_pair_confidence.get(key, 0) + 1

        # Word Trigrams
        if num_tokens >= 3:
            for i in range(num_tokens - 2):
                w1, w2, w3 = raw_words[i], raw_words[i+1], raw_words[i+2]
                if ',' in [w1, w2, w3]: continue
                key = f"{w1},{w2},{w3}"
                word_trigram_raw_scores[key] = word_trigram_raw_scores.get(key, 0.0) + (sentence_score * 2.0)
                word_trigram_confidence[key] = word_trigram_confidence.get(key, 0) + 1

        # Endings
        for i in reversed(range(num_tokens)):
            if raw_words[i] != ',':
                last_pos = pos_sequence[i]
                pos_ending_raw_scores[last_pos] = pos_ending_raw_scores.get(last_pos, 0.0) + sentence_score
                pos_ending_confidence[last_pos] = pos_ending_confidence.get(last_pos, 0) + 1
                break

    def normalize(raw_dict, conf_dict):
        final = {}
        max_abs = 0.0
        temp = {}
        for k, score in raw_dict.items():
            conf = conf_dict[k]
            if conf >= min_confidence_count:
                eff = score * math.log(conf + 1)
                temp[k] = eff
                max_abs = max(max_abs, abs(eff))
        if max_abs > 0:
            for k, eff in temp.items():
                if abs(eff) < ignore_threshold: continue
                final[k] = (eff / max_abs + 1) / 2
        return final

    # Filter vocab to top keywords
    sorted_vocab = sorted(theme_vocab.items(), key=lambda x: x[1], reverse=True)
    top_vocab = [word for word, score in sorted_vocab[:500]]

    return {
        "pos_adjacency": normalize(pos_pair_raw_scores, pos_pair_confidence),
        "word_trigrams": normalize(word_trigram_raw_scores, word_trigram_confidence),
        "pos_endings": normalize(pos_ending_raw_scores, pos_ending_confidence),
        "vocab": top_vocab
    }

def run_learner(input_csv_path, output_json_path, my_quotes_dir):
    theme_data = {}
    
    # 1. User ratings
    if os.path.exists(input_csv_path):
        theme_data['user'] = []
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rating = row['rating'].strip().lower()
                if rating == 'no_rating': continue
                score = 1.0 if rating == '+' else -1.0
                words = []
                for t in row['quote'].replace('.', '').split():
                    if t.endswith(','): words.extend([t[:-1], ','])
                    else: words.append(t)
                theme_data['user'].append({'raw_words': words, 'sentence_score': score})

    # 2. Themes from folders
    if os.path.exists(my_quotes_dir):
        for entry in os.scandir(my_quotes_dir):
            if entry.is_dir():
                theme_name = entry.name
                theme_data[theme_name] = theme_data.get(theme_name, [])
                for filename in os.listdir(entry.path):
                    if filename.endswith(".txt"):
                        with open(os.path.join(entry.path, filename), 'r', encoding='utf-8') as f:
                            for line in f:
                                q = line.strip()
                                if not q: continue
                                words = []
                                for t in q.replace('.', '').split():
                                    if t.endswith(','): words.extend([t[:-1], ','])
                                    else: words.append(t)
                                theme_data[theme_name].append({'raw_words': words, 'sentence_score': 2.0})

    final_output = {"themes": {}}
    for theme_name, quotes in theme_data.items():
        if not quotes: continue
        print(f"Refining theme: {theme_name}...")
        final_output["themes"][theme_name] = process_quote_list(quotes, 2, 0.05)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    run_learner("data/quotes_for_learning.csv", "models/learned_parameters.json", "data/my_quotes")
