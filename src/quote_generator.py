import random
import re
import argparse
import uuid
import os
import json
import csv # Import csv module

# Import POS tagging logic and data from pos_tagger.py
from pos_tagger import all_words, assign_role_by_pattern, FUNC, PRON, MODAL, DET


# --- Learned Parameters (Initially empty, to be loaded) ---
LEARNED_POS_ADJACENCY_SCORES = {}
LEARNED_WORD_ADJACENCY_SCORES = {}
LEARNED_POS_ENDING_PROBABILITIES = {}
LEARNED_WORD_ENDING_PROBABILITIES = {}

def load_learned_parameters(filepath="learned_parameters.json"):
    global LEARNED_POS_ADJACENCY_SCORES, LEARNED_WORD_ADJACENCY_SCORES
    global LEARNED_POS_ENDING_PROBABILITIES, LEARNED_WORD_ENDING_PROBABILITIES
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
                LEARNED_POS_ADJACENCY_SCORES = {k: v for k, v in params.get("pos_adjacency_scores", {}).items()}
                LEARNED_WORD_ADJACENCY_SCORES = {k: v for k, v in params.get("word_adjacency_scores", {}).items()}
                LEARNED_POS_ENDING_PROBABILITIES = params.get("pos_ending_probabilities", {})
                LEARNED_WORD_ENDING_PROBABILITIES = params.get("word_ending_probabilities", {})
        except json.JSONDecodeError:
            print(f"Error decoding learned parameters from {filepath}. Using default.")
        except Exception as e:
            print(f"An unexpected error occurred loading learned parameters: {e}. Using default.")


# --- Adjacency Logic ---
# Probabilities for restricted pairs (0 to 1)
RESTRICTED_PROB = 0.2  # 20% chance for restricted pairs
VING_VING_RESTRICTED_PROB = 0.1 # Even lower chance for VING + VING

def check_adjacency(prev_word, prev_pos, current_word, current_pos, exploration_rate): # Added exploration_rate
    # Rule: Absolute forbidden POS rules (these are ALWAYS active, cannot be overridden by learning or exploration)
    if (prev_pos == 'FUNC' and prev_word and prev_word.lower() == 'to' and current_pos == 'VING') or \
       (prev_pos == 'MODAL' and current_pos == 'VING') or \
       (prev_pos == 'VING' and current_pos == 'MODAL') or \
       (prev_pos == 'FUNC' and current_pos in ['FUNC', 'MODAL']) or \
       (prev_pos == 'MODAL' and current_pos in ['FUNC', 'MODAL']) or \
       (prev_pos == 'PRON' and current_pos == 'PRON') or \
       (prev_pos == 'DET' and current_pos == 'DET'):
        return False, 0.0

    # If exploring, bypass learned parameters and go straight to heuristics
    if random.random() < exploration_rate:
        # Fall through to original heuristic rules below
        pass
    else:
        # --- Priority 1: Learned Word-level Adjacency Score ---
        word_pair_key_str = f"{prev_word},{current_word}"
        if word_pair_key_str in LEARNED_WORD_ADJACENCY_SCORES:
            score = LEARNED_WORD_ADJACENCY_SCORES[word_pair_key_str]
            if score <= 0: return False, 0.0
            return True, max(0.01, min(1.0, score))

        # --- Priority 2: Learned POS-level Adjacency Score ---
        pos_pair_key_str = f"{prev_pos},{current_pos}"
        if pos_pair_key_str in LEARNED_POS_ADJACENCY_SCORES:
            score = LEARNED_POS_ADJACENCY_SCORES[pos_pair_key_str]
            if score <= 0: return False, 0.0
            return True, max(0.01, min(1.0, score))


    # --- Priority 3 (or fallback during exploration): Original heuristic POS rules and default probabilities ---
    if (prev_pos == 'NOUN' and current_pos == 'NOUN') or \
       (prev_pos == 'ADJ' and current_pos == 'ADJ') or \
       (prev_pos == 'ADV' and current_pos == 'ADV'):
        return True, RESTRICTED_PROB
    if prev_pos == 'VING' and current_pos == 'VING':
        return True, VING_VING_RESTRICTED_PROB

    # Preferred pairs (always 1.0 prob)
    if (prev_pos == 'ADJ' and current_pos == 'NOUN') or \
       (prev_pos == 'VING' and current_pos == 'NOUN') or \
       (prev_pos == 'ADV' and current_pos == 'ADJ') or \
       (prev_pos == 'NOUN' and current_pos == 'VING'):
        return True, 1.0

    # All other pairs get a default probability
    return True, 0.5


# --- Ending Logic ---
def check_ending_viability(last_word, last_pos, exploration_rate): # Added exploration_rate
    # Rule: Absolute forbidden POS endings (these are ALWAYS active, cannot be overridden by learning or exploration)
    if last_pos in ['FUNC', 'MODAL', 'PRON', 'DET']:
        return False, 0.0

    # If exploring, bypass learned parameters and go straight to heuristics
    if random.random() < exploration_rate:
        # Fall through to original heuristic rules below
        pass
    else:
        # --- Priority 1: Learned Word-level Ending Probability ---
        if last_word in LEARNED_WORD_ENDING_PROBABILITIES:
            prob = LEARNED_WORD_ENDING_PROBABILITIES[last_word]
            return True, max(0.01, min(1.0, prob))

        # --- Priority 2: Learned POS-level Ending Probability ---
        if last_pos in LEARNED_POS_ENDING_PROBABILITIES:
            prob = LEARNED_POS_ENDING_PROBABILITIES[last_pos]
            return True, max(0.01, min(1.0, prob)) # Cap probability

    # --- Priority 3 (or fallback during exploration): Original heuristic POS rules and default probabilities ---
    if last_pos in ['VERB', 'VING', 'ADV']:
        return True, 0.3

    if last_pos in ['NOUN', 'ADJ']:
        return True, 1.0

    return False, 0.0 # Should not be reached


# --- Phrase Generation Logic ---
def generate_phrase(min_length_words=3, max_length_words=10, max_attempts_per_word=20, is_second_part=False, exploration_rate=0.0): # Added exploration_rate
    phrase = []
    prev_word = None
    prev_pos = None
    word_pool_list = list(all_words) # all_words is from pos_tagger.py

    # To prevent infinite recursion in extreme cases where rules are too strict for a small pool
    current_recursion_depth = 0
    MAX_RECURSION_DEPTH = 5

    while True:
        current_word = None
        current_pos = None
        attempt_count = 0

        while attempt_count < max_attempts_per_word:
            attempt_count += 1
            word_candidate = random.choice(word_pool_list)

            if prev_word is not None and word_candidate == prev_word:
                continue

            candidate_pos = assign_role_by_pattern(word_candidate) # assign_role_by_pattern is from pos_tagger.py

            # Rule for the very first word of this phrase/segment
            if not phrase:
                # If it's the start of the whole quote or the start of the second part (after comma)
                # Avoid starting with FUNC, MODAL, DET
                if candidate_pos in ['FUNC', 'MODAL', 'DET']:
                    continue
            
            # Adjacency check
            if prev_pos is not None:
                is_allowed, probability = check_adjacency(prev_word, prev_pos, word_candidate, candidate_pos, exploration_rate) # Pass exploration_rate
                if not is_allowed:
                    continue
                if random.random() > probability:
                    continue

            current_word = word_candidate
            current_pos = candidate_pos
            break

        if current_word is None:
            # Backtrack
            if phrase:
                phrase.pop()
                prev_word = phrase[-1] if phrase else None
                prev_pos = assign_role_by_pattern(prev_word) if prev_word else None
                continue
            else:
                # If we couldn't even start, rules are too strict. Indicate failure or restart.
                current_recursion_depth += 1
                if current_recursion_depth > MAX_RECURSION_DEPTH:
                    raise RecursionError("Max phrase generation recursion depth reached. Rules too strict or word pool too small.")
                return generate_phrase(min_length_words, max_length_words, max_attempts_per_word, is_second_part, exploration_rate) # Pass exploration_rate

        phrase.append(current_word)
        prev_word = current_word
        prev_pos = assign_role_by_pattern(prev_word)

        # Check if we should end this phrase/segment
        if len(phrase) >= min_length_words:
            is_end_viable, prob_to_end = check_ending_viability(prev_word, prev_pos, exploration_rate) # Pass exploration_rate
            if is_end_viable and random.random() < prob_to_end:
                return phrase # Return raw word list
            elif len(phrase) >= max_length_words:
                # Force end if max length reached, try to make it viable if possible
                if not is_end_viable and prob_to_end == 0.0:
                    # If strictly forbidden to end here, try backtracking once to find an end.
                    if phrase:
                        phrase.pop()
                        prev_word = phrase[-1] if phrase else None
                        prev_pos = assign_role_by_pattern(prev_word) if prev_word else None
                        # Try to find a word that allows ending from this new state
                        temp_phrase_start = list(phrase) # Base for new sub-generation
                        temp_prev_word = prev_word
                        temp_prev_pos = assign_role_by_pattern(temp_prev_word) if temp_prev_word else None
                        found_ending_word = False
                        for _ in range(max_attempts_per_word):
                            end_word_candidate = random.choice(word_pool_list)
                            if temp_prev_word is not None and end_word_candidate == temp_prev_word:
                                continue
                            end_candidate_pos = assign_role_by_pattern(end_word_candidate)
                            is_allowed, probability = check_adjacency(temp_prev_word, temp_prev_pos, end_word_candidate, end_candidate_pos, exploration_rate) # Pass exploration_rate
                            if is_allowed and random.random() < probability:
                                end_viable, _ = check_ending_viability(end_word_candidate, end_candidate_pos, exploration_rate) # Pass exploration_rate
                                if end_viable:
                                    temp_phrase_start.append(end_word_candidate)
                                    found_ending_word = True
                                    return temp_phrase_start # Return phrase with found ending
                        
                        if not found_ending_word:
                            # If failed to find a good ending word after backtracking
                            current_recursion_depth += 1
                            if current_recursion_depth > MAX_RECURSION_DEPTH:
                                raise RecursionError("Max phrase generation recursion depth reached. Rules too strict or word pool too small.")
                            return generate_phrase(min_length_words, max_length_words, max_attempts_per_word, is_second_part, exploration_rate) # Pass exploration_rate
                    else: # Phrase became empty after pop, restart
                        current_recursion_depth += 1
                        if current_recursion_depth > MAX_RECURSION_DEPTH:
                            raise RecursionError("Max phrase generation recursion depth reached. Rules too strict or word pool too small.")
                        return generate_phrase(min_length_words, max_length_words, max_attempts_per_word, is_second_part, exploration_rate) # Pass exploration_rate
                else: # Ending is restricted or free, just end it at max_length
                    return phrase

# --- Full Quote Generation Wrapper ---
def generate_full_quote(min_total_length=8, max_total_length=20, two_part_prob=0.6, return_raw_words=False, exploration_rate=0.0): # Added exploration_rate
    # No seed setting here, it's handled at the top level
    
    # Ensure minimum lengths are reasonable
    min_total_length = max(min_total_length, 6) # At least 6 words for a meaningful quote, especially two-part
    max_total_length = max(max_total_length, min_total_length + 2) # Ensure max is at least min + buffer

    # Try generating up to 3 times for a full quote if specific part generation fails
    for _ in range(3):
        try:
            is_two_part = random.random() < two_part_prob
            
            if is_two_part:
                # Generate two parts
                min_len1 = max(3, int(min_total_length / 2))
                max_len1 = max(min_len1, int(max_total_length * 0.6) - 1)

                part1 = generate_phrase(min_length_words=min_len1, max_length_words=max_len1, is_second_part=False, exploration_rate=exploration_rate) # Pass exploration_rate
                
                remaining_length_for_part2 = max_total_length - len(part1) - 1
                min_len2 = max(3, min_total_length - len(part1) - 1)
                
                part2 = generate_phrase(min_length_words=min_len2, max_length_words=remaining_length_for_part2, is_second_part=True, exploration_rate=exploration_rate) # Pass exploration_rate
                
                full_quote_words = part1 + [','] + part2
            else:
                # Generate single part
                full_quote_words = generate_phrase(min_length_words=min_total_length, max_length_words=max_total_length, is_second_part=False, exploration_rate=exploration_rate) # Pass exploration_rate
            
            final_quote_text = ' '.join(full_quote_words).replace(' ,', ',').capitalize() + '.'

            if return_raw_words:
                return final_quote_text, full_quote_words # Return raw word list for learner
            else:
                return final_quote_text
        except RecursionError:
            # If a part generation failed, try the whole full quote generation again
            continue # Loop will re-attempt
    
    # If all attempts fail, return a fallback message
    if return_raw_words:
        return "Could not generate a quote due to strict rules or limited word pool.", []
    else:
        return "Could not generate a quote due to strict rules or limited word pool."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate grammatically structured quotes.")
    parser.add_argument("--learn", action="store_true", help="Log generated quotes to a file for learning.")
    parser.add_argument("--seed", type=int, help="Provide an integer seed for reproducible quote generation.")
    parser.add_argument("--num_quotes", type=int, default=5, help="Number of quotes to generate.")
    parser.add_argument("--learned_params", type=str, default="models/learned_parameters.json", 
                        help="Path to learned parameters JSON file.")
    parser.add_argument("--exploration_rate", type=float, default=0.1, 
                        help="Probability (0.0 to 1.0) to ignore learned parameters and use heuristics for novelty.")
    parser.add_argument("--rate", action="store_true", 
                        help="Generate quotes and prompt for immediate interactive rating.") # Changed flag name
    parser.add_argument("--raw", action="store_true", help="Print only the quote text without labels or headers.")
    args = parser.parse_args()

    # Load learned parameters if they exist
    load_learned_parameters(args.learned_params)

    # Set the seed once for the entire sequence of quotes if provided
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(None) # Initialize with system time for randomness

    LOG_FILE_PATH = "data/quotes_for_learning.csv"
    
    # Mode: Interactive Rating
    if args.rate: # Changed check
        print(f"\n--- Interactive Quote Rating Mode ---")
        print(f"Quotes will be logged to {LOG_FILE_PATH}")
        file_exists = os.path.exists(LOG_FILE_PATH) and os.stat(LOG_FILE_PATH).st_size > 0
        with open(LOG_FILE_PATH, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            if not file_exists:
                csv_writer.writerow(["id", "quote", "rating"]) # Write header only if file is new/empty

            for i in range(args.num_quotes):
                quote_id = uuid.uuid4()
                quote_text, raw_words = generate_full_quote(min_total_length=8, max_total_length=20, two_part_prob=0.6, 
                                                            return_raw_words=True, exploration_rate=args.exploration_rate) # Pass exploration_rate
                
                print(f"\nQuote {i+1} ({quote_id}): {quote_text}")
                user_input = input("Rate this quote (+/- to rate, Enter to skip, q to quit): ").strip().lower() # Added q to prompt and lower()
                
                if user_input == 'q': # Check for quit command
                    print("Quitting interactive rating.")
                    break # Exit the loop
                
                rating_to_log = "NO_RATING"
                if user_input == '+':
                    rating_to_log = '+'
                elif user_input == '-':
                    rating_to_log = '-'
                
                csv_writer.writerow([str(quote_id), quote_text, rating_to_log])
                print(f"Rating '{rating_to_log}' recorded.")
        print("\n--- Interactive Rating Complete ---")
        print(f"Run 'python3 learner.py --input_csv {LOG_FILE_PATH}' to update learned parameters.")
    
    # Mode: Batch Logging (original --learn functionality)
    elif args.learn: # This will be exclusive with --rate_interactive
        print(f"Logging quotes to {LOG_FILE_PATH} for learning...")
        file_exists = os.path.exists(LOG_FILE_PATH) and os.stat(LOG_FILE_PATH).st_size > 0
        with open(LOG_FILE_PATH, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            if not file_exists:
                csv_writer.writerow(["id", "quote", "rating"]) # Write header only if file is new/empty

            print(f"\nGenerating {args.num_quotes} quotes:")
            for i in range(args.num_quotes):
                quote_id = uuid.uuid4()
                quote_text, raw_words = generate_full_quote(min_total_length=8, max_total_length=20, two_part_prob=0.6, 
                                                            return_raw_words=True, exploration_rate=args.exploration_rate) # Pass exploration_rate
                
                csv_writer.writerow([str(quote_id), quote_text, "NO_RATING"])
                print(f"Quote {i+1} ({quote_id}): {quote_text}")
    
    # Default Mode: Just Generate and Print (no logging)
    else:
        if not args.raw:
            print(f"\nGenerating {args.num_quotes} quotes:")
        for i in range(args.num_quotes):
            quote_text = generate_full_quote(min_total_length=8, max_total_length=20, two_part_prob=0.6, exploration_rate=args.exploration_rate) # Pass exploration_rate
            if args.raw:
                print(quote_text)
            else:
                print(f"Quote {i+1}: {quote_text}")