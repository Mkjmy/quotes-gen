import random
import argparse
import os
import json
import subprocess
from datetime import datetime

from pos_tagger import all_words, assign_role_by_pattern

LEARNED_DATA = {"themes": {}}
CURRENT_THEME = "general"

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

def generate_phrase(min_len=4, max_len=12, drift_chance=0.2):
    theme_dict = get_theme_dict()
    vocab = theme_dict.get("vocab", [])
    trigrams = theme_dict.get("word_trigrams", {})
    pos_adj = theme_dict.get("pos_adjacency", {})
    pos_ends = theme_dict.get("pos_endings", {})

    phrase = []
    w_prev1, w_prev2 = None, None

    for i in range(max_len):
        # 1. Candidate Selection
        candidates = []
        
        # Priority A: Trigram (from learned data)
        if w_prev2 and w_prev1:
            prefix = f"{w_prev2},{w_prev1},"
            tri_matches = [k.split(',')[-1] for k in trigrams.keys() if k.startswith(prefix)]
            candidates.extend(tri_matches * 5) # Weight trigrams heavily

        # Priority B: Philosophical Vocab (Gravity)
        if not candidates or random.random() < 0.3:
            candidates.extend(random.sample(vocab, min(len(vocab), 20)))

        # Priority C: The Leap (Drift into the unknown/vo tri)
        if random.random() < drift_chance:
            candidates.extend(random.sample(list(all_words), 10))

        if not candidates: candidates = random.sample(list(all_words), 5)

        # 2. Validation & Scoring
        best_word = None
        best_score = -1
        
        random.shuffle(candidates)
        for cand in candidates[:30]:
            if cand == w_prev1: continue
            
            c_pos = assign_role_by_pattern(cand)
            p1_pos = assign_role_by_pattern(w_prev1) if w_prev1 else None
            
            # POS logic check
            score = 0.5
            if p1_pos:
                pos_key = f"{p1_pos},{c_pos}"
                score = pos_adj.get(pos_key, 0.3)
                # Hard rules
                if p1_pos == 'FUNC' and c_pos == 'FUNC': score *= 0.1
                if p1_pos == c_pos and p1_pos in ['PRON', 'DET', 'MODAL']: score *= 0.0
            
            # Start word check
            if not phrase and c_pos in ['FUNC', 'MODAL', 'DET']: score *= 0.1

            if score > best_score:
                best_score = score
                best_word = cand
            if best_score > 0.8 and random.random() < 0.7: break

        if not best_word: break
        
        phrase.append(best_word)
        w_prev2, w_prev1 = w_prev1, best_word

        # 3. Dynamic Ending
        if len(phrase) >= min_len:
            last_pos = assign_role_by_pattern(phrase[-1])
            end_prob = pos_ends.get(last_pos, 0.1)
            if last_pos in ['FUNC', 'DET', 'MODAL']: end_prob = 0
            
            if random.random() < end_prob:
                break
                
    return phrase

def generate_full_quote(theme="general"):
    global CURRENT_THEME
    CURRENT_THEME = theme
    
    # Sometimes a single flow, sometimes a complex one with a comma
    if random.random() < 0.6:
        p1 = generate_phrase(min_len=4, max_len=8, drift_chance=0.15)
        p2 = generate_phrase(min_len=3, max_len=6, drift_chance=0.3)
        full = p1 + [","] + p2
    else:
        full = generate_phrase(min_len=6, max_len=12, drift_chance=0.2)
        
    return " ".join(full).replace(" ,", ",").capitalize() + "."

def generate_paragraph(theme="general", num_sentences=8):
    transitions = ["Moreover, ", "Thus, ", "Yet, ", "In the end, ", "Therefore, ", "Consequently, ", "Simply put, ", "Indeed, ", "Beyond this, "]
    paragraph = []
    for i in range(num_sentences):
        sentence = generate_full_quote(theme=theme)
        if i > 0 and random.random() < 0.5:
            trans = random.choice(transitions)
            sentence = trans + sentence[0].lower() + sentence[1:]
        paragraph.append(sentence)
    return " ".join(paragraph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_quotes", type=int, default=5)
    parser.add_argument("--theme", default="general")
    parser.add_argument("--paragraph", action="store_true")
    parser.add_argument("--sentences", type=int, default=8, help="Number of sentences in the paragraph")
    parser.add_argument("--svg", action="store_true", help="Export to a stylish SVG image")
    parser.add_argument("--image", action="store_true", help="Export to a professional PNG image (requires venv)")
    args = parser.parse_args()

    load_learned_parameters()
    
    # Verify theme
    if args.theme not in LEARNED_DATA.get("themes", {}):
        args.theme = "general" if "general" in LEARNED_DATA.get("themes", {}) else list(LEARNED_DATA.get("themes", {}).keys())[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.paragraph:
        final_text = generate_paragraph(theme=args.theme, num_sentences=args.sentences)
        print(f"\n--- [THEME: {args.theme.upper()}] DEEP PERSPECTIVE ---\n")
        print(final_text)
        print("\n" + "="*60 + "\n")
        
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
            final_text = generate_full_quote(theme=args.theme)
            q_id = str(uuid.uuid4())
            print(f"[{args.theme.upper()}] {final_text}")
            
            base_name = f"quote_{args.theme.upper()}_{timestamp}_{i+1}"
            if args.svg:
                from svg_generator import generate_svg
                generate_svg(final_text, theme=args.theme.upper(), output_path=f"{base_name}.svg")
            if args.image:
                out_path = os.path.join(output_dir, f"{base_name}.png")
                if os.path.exists(venv_python):
                    subprocess.run([venv_python, "src/image_generator.py", final_text, args.theme.upper(), out_path, q_id])
