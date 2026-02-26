import argparse
import csv
import json
import os
import math

# Import POS tagging logic and data from pos_tagger.py
from pos_tagger import all_words, assign_role_by_pattern, FUNC, PRON, MODAL, DET # Import specific constants for heuristic checks


# --- Helper Function for Learning Principles ---

# Updated get_pair_weight to take raw_words_list directly
def get_pair_weight(index, total_tokens, raw_words_list):
    """
    Calculates the positional weight for a word pair (raw_words_list[index], raw_words_list[index+1])
    based on its position in the quote and proximity to commas.
    Implements "Thưởng theo vị trí" (Reward by position).
    """
    if total_tokens <= 1: # Avoid division by zero or nonsensical weights for very short quotes
        return 0.0

    # Weights for start/end of the sequence of tokens
    if index == 0 or index == total_tokens - 2: # The pair is (raw_words_list[index], raw_words_list[index+1])
        return 1.0        # đầu / cuối
    
    # Weights for near comma
    is_near_comma = False
    # Check if any token in the pair itself is a comma
    if raw_words_list[index] == ',' or raw_words_list[index+1] == ',':
        is_near_comma = True
    # Check if the token immediately before the pair is a comma
    if index > 0 and raw_words_list[index-1] == ',':
        is_near_comma = True
    # Check if the token immediately after the pair is a comma
    if index + 2 < total_tokens and raw_words_list[index+2] == ',':
        is_near_comma = True

    if is_near_comma:
        return 0.8
        
    return 0.3            # giữa câu

def violates_heuristic(prev_pos, current_pos, prev_word):
    """
    Checks if a pair violates known bad heuristic rules.
    This should align with the absolute forbidden rules in quote_generator.py.
    """
    # Align with check_adjacency's absolute forbidden pairs
    # Note: FUNC, PRON, MODAL, DET are imported from pos_tagger.py
    if (
        (prev_pos == 'FUNC' and prev_word and prev_word.lower() == 'to' and current_pos == 'VING') or
        (prev_pos == 'MODAL' and current_pos == 'VING') or
        (prev_pos == 'VING' and current_pos == 'MODAL') or
        (prev_pos == 'FUNC' and current_pos in ['FUNC', 'MODAL']) or
        (prev_pos == 'MODAL' and current_pos in ['FUNC', 'MODAL']) or
        (prev_pos == 'PRON' and current_pos == 'PRON') or
        (prev_pos == 'DET' and current_pos == 'DET')
    ):
        return True
    return False

# --- Main Learner Logic ---
def run_learner(input_csv_path, output_json_path, default_score_good, default_score_bad,
                min_confidence_count, ignore_threshold, base_penalty_for_bad_heuristic):

    # Initialize data structures for POS-level learning
    pos_pair_raw_scores = {}
    pos_pair_confidence = {}
    pos_ending_raw_scores = {}
    pos_ending_confidence = {}

    # Initialize data structures for Word-level learning
    word_pair_raw_scores = {}
    word_pair_confidence = {}
    word_ending_raw_scores = {}
    word_ending_confidence = {}

    processed_quotes = []

    # Load and parse CSV
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return

    # Read all quotes from the CSV, will reorder later
    all_quotes_from_csv_raw = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_quotes_from_csv_raw.append(row)

    for row in all_quotes_from_csv_raw:
        quote_id = row['id']
        quote_text = row['quote']
        rating_str = row['rating'].strip().lower()

        if rating_str == 'no_rating':
            continue # Skip unrated quotes for learning calculations
        
        # Convert rating to numeric score based on '+' or '-'
        sentence_score = 0
        if rating_str == '+':
            sentence_score = default_score_good
        elif rating_str == '-':
            sentence_score = default_score_bad
        else:
            print(f"Warning: Unknown rating '{rating_str}' for quote ID {quote_id}. Skipping for learning calculations.")
            continue
        
        # Tokenize quote text to get raw words (handling comma)
        cleaned_text = quote_text.replace('.', '') # Remove final period
        raw_words_list = []
        for token in cleaned_text.split():
            if token.endswith(','):
                raw_words_list.append(token[:-1])
                raw_words_list.append(',')
            else:
                raw_words_list.append(token)

        if not raw_words_list:
            continue

        processed_quotes.append({
            'id': quote_id,
            'raw_words': raw_words_list,
            'sentence_score': sentence_score
        })

    # --- Iterate through rated quotes and apply learning principles ---
    for quote_data in processed_quotes:
        raw_words = quote_data['raw_words']
        sentence_score = quote_data['sentence_score']

        pos_sequence = []
        for word in raw_words:
            if word == ',': # Explicitly assign 'COMMA' as POS for comma token
                pos_sequence.append('COMMA')
            else:
                pos_sequence.append(assign_role_by_pattern(word))
        
        num_tokens = len(raw_words)
        reward_per_pair_base = sentence_score / (num_tokens - 1) if num_tokens > 1 else 0

        # Process pairs (both POS-level and Word-level)
        for i in range(num_tokens - 1):
            prev_word = raw_words[i]
            prev_pos = pos_sequence[i]
            current_word = raw_words[i+1]
            current_pos = pos_sequence[i+1]

            # Skip learning from comma-related pairs for now
            if prev_pos == 'COMMA' or current_pos == 'COMMA':
                continue

            # --- POS-level learning ---
            pos_pair_key_str = f"{prev_pos},{current_pos}"

            pair_weight = get_pair_weight(i, num_tokens, raw_words) # Use updated get_pair_weight
            current_reward_pos = reward_per_pair_base * pair_weight

            if violates_heuristic(prev_pos, current_pos, prev_word): # Heuristics primarily POS-based
                current_reward_pos -= base_penalty_for_bad_heuristic

            pos_pair_raw_scores[pos_pair_key_str] = pos_pair_raw_scores.get(pos_pair_key_str, 0.0) + current_reward_pos
            pos_pair_confidence[pos_pair_key_str] = pos_pair_confidence.get(pos_pair_key_str, 0) + 1

            # --- Word-level learning ---
            word_pair_key_str = f"{prev_word},{current_word}" # Key now includes actual words

            pair_weight = get_pair_weight(i, num_tokens, raw_words) # Use updated get_pair_weight
            current_reward_word = reward_per_pair_base * pair_weight

            # Optionally, apply heuristic penalty for word-level if there are any specific word-level bad heuristics
            # For now, we'll reuse the POS-level heuristic check, which is fine as it mainly checks POS rules
            if violates_heuristic(prev_pos, current_pos, prev_word): 
                 current_reward_word -= base_penalty_for_bad_heuristic

            word_pair_raw_scores[word_pair_key_str] = word_pair_raw_scores.get(word_pair_key_str, 0.0) + current_reward_word
            word_pair_confidence[word_pair_key_str] = word_pair_confidence.get(word_pair_key_str, 0) + 1
        
        # Process ending POS (both POS-level and Word-level)
        last_non_comma_word = None
        last_non_comma_pos = None
        for i in reversed(range(num_tokens)):
            if raw_words[i] != ',':
                last_non_comma_word = raw_words[i]
                last_non_comma_pos = pos_sequence[i]
                break

        if last_non_comma_pos and last_non_comma_pos != 'COMMA':
            # POS-level ending
            pos_ending_raw_scores[last_non_comma_pos] = pos_ending_raw_scores.get(last_non_comma_pos, 0.0) + sentence_score
            pos_ending_confidence[last_non_comma_pos] = pos_ending_confidence.get(last_non_comma_pos, 0) + 1

            # Word-level ending
            word_ending_raw_scores[last_non_comma_word] = word_ending_raw_scores.get(last_non_comma_word, 0.0) + sentence_score
            word_ending_confidence[last_non_comma_word] = word_ending_confidence.get(last_non_comma_word, 0) + 1


    # --- Calculate Effective Scores and Probabilities for POS-level ---
    learned_pos_adjacency_scores = {}
    max_abs_pos_pair_score = 0.0

    for pk_str, raw_score in pos_pair_raw_scores.items():
        conf = pos_pair_confidence[pk_str]
        if conf >= min_confidence_count:
            effective_score = raw_score * math.log(conf + 1)
            learned_pos_adjacency_scores[pk_str] = effective_score
            max_abs_pos_pair_score = max(max_abs_pos_pair_score, abs(effective_score))
    
    final_pos_adjacency_probabilities = {}
    if max_abs_pos_pair_score > 0:
        for pk_str, eff_score in learned_pos_adjacency_scores.items():
            normalized_score = eff_score / max_abs_pos_pair_score
            prob = (normalized_score + 1) / 2
            if abs(eff_score) < ignore_threshold:
                continue
            final_pos_adjacency_probabilities[pk_str] = prob
            
    learned_pos_ending_probabilities = {}
    max_abs_pos_ending_score = 0.0

    for pos, raw_score in pos_ending_raw_scores.items():
        conf = pos_ending_confidence[pos]
        if conf >= min_confidence_count:
            effective_score = raw_score * math.log(conf + 1)
            learned_pos_ending_probabilities[pos] = effective_score
            max_abs_pos_ending_score = max(max_abs_pos_ending_score, abs(effective_score))
            
    if max_abs_pos_ending_score > 0:
        for pos, eff_score in learned_pos_ending_probabilities.items():
            normalized_score = eff_score / max_abs_pos_ending_score
            prob = (normalized_score + 1) / 2
            if abs(eff_score) < ignore_threshold:
                continue
            learned_pos_ending_probabilities[pos] = prob

    # --- Calculate Effective Scores and Probabilities for Word-level ---
    learned_word_adjacency_scores = {}
    max_abs_word_pair_score = 0.0

    for wpk_str, raw_score in word_pair_raw_scores.items():
        conf = word_pair_confidence[wpk_str]
        if conf >= min_confidence_count:
            effective_score = raw_score * math.log(conf + 1)
            learned_word_adjacency_scores[wpk_str] = effective_score
            max_abs_word_pair_score = max(max_abs_word_pair_score, abs(effective_score))
    
    final_word_adjacency_probabilities = {}
    if max_abs_word_pair_score > 0:
        for wpk_str, eff_score in learned_word_adjacency_scores.items():
            normalized_score = eff_score / max_abs_word_pair_score
            prob = (normalized_score + 1) / 2
            if abs(eff_score) < ignore_threshold:
                continue
            final_word_adjacency_probabilities[wpk_str] = prob

    learned_word_ending_probabilities = {}
    max_abs_word_ending_score = 0.0

    for word, raw_score in word_ending_raw_scores.items():
        conf = word_ending_confidence[word]
        if conf >= min_confidence_count:
            effective_score = raw_score * math.log(conf + 1)
            learned_word_ending_probabilities[word] = effective_score
            max_abs_word_ending_score = max(max_abs_word_ending_score, abs(effective_score))
            
    if max_abs_word_ending_score > 0:
        for word, eff_score in learned_word_ending_probabilities.items():
            normalized_score = eff_score / max_abs_word_ending_score
            prob = (normalized_score + 1) / 2
            if abs(eff_score) < ignore_threshold:
                continue
            learned_word_ending_probabilities[word] = prob


    # --- Output to JSON ---
    output_params = {
        "pos_adjacency_scores": final_pos_adjacency_probabilities,
        "word_adjacency_scores": final_word_adjacency_probabilities,
        "pos_ending_probabilities": learned_pos_ending_probabilities,
        "word_ending_probabilities": learned_word_ending_probabilities
    }
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_params, f, indent=4)
    
    print(f"\nLearned parameters saved to {output_json_path}")
    print(f"  Learned POS adjacency pairs: {len(final_pos_adjacency_probabilities)}")
    print(f"  Learned Word adjacency pairs: {len(final_word_adjacency_probabilities)}")
    print(f"  Learned POS ending probabilities: {len(learned_pos_ending_probabilities)}")
    print(f"  Learned Word ending probabilities: {len(learned_word_ending_probabilities)}")

    # --- Reorder quotes_for_learning.csv ---
    print("\nReordering quotes_for_learning.csv: Unrated quotes will be moved to the top.")
    
    unrated_quotes = []
    rated_quotes = []
    
    # Read all quotes from the CSV
    all_quotes_from_csv = []
    if os.path.exists(input_csv_path):
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_quotes_from_csv.append(row)

    for row in all_quotes_from_csv:
        if row['rating'].strip().lower() == 'no_rating':
            unrated_quotes.append(row)
        else:
            rated_quotes.append(row)
    
    # Combine (unrated first, then rated)
    reordered_quotes = unrated_quotes + rated_quotes

    # Rewrite the CSV file
    if reordered_quotes: # Only write if there are quotes to write
        with open(input_csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['id', 'quote', 'rating'] # Ensure fieldnames are correct
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reordered_quotes)
        print(f"Reordered {len(unrated_quotes)} unrated quotes to the top of {input_csv_path}.")
    else:
        print(f"No quotes found in {input_csv_path} to reorder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze rated quotes and learn parameters for quote generation.")
    parser.add_argument("--input_csv", type=str, default="data/quotes_for_learning.csv",
                        help="Path to the CSV file containing quotes and ratings.")
    parser.add_argument("--output_json", type=str, default="models/learned_parameters.json",
                        help="Path to save the learned parameters JSON file.")
    parser.add_argument("--score_good", type=float, default=1.0,
                        help="Numeric score for a 'good' rating.")
    parser.add_argument("--score_bad", type=float, default=-1.0,
                        help="Numeric score for a 'bad' rating.")
    parser.add_argument("--min_confidence_count", type=int, default=3,
                        help="Minimum number of occurrences (confidence) for a pair/POS to be learned.")
    parser.add_argument("--ignore_threshold", type=float, default=0.1,
                        help="Absolute effective score threshold below which a pair/POS is ignored.")
    parser.add_argument("--base_penalty_bad_heuristic", type=float, default=0.5,
                        help="Penalty applied to reward if a pair violates known bad heuristics.")
    
    args = parser.parse_args()

    run_learner(
        args.input_csv,
        args.output_json,
        args.score_good,
        args.score_bad,
        args.min_confidence_count,
        args.ignore_threshold,
        args.base_penalty_bad_heuristic
    )
