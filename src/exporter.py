import csv
import os
import subprocess

def export_all_quotes(csv_path="data/quotes_for_learning.csv", output_dir="output_images"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if not os.path.exists(venv_python):
        print("Error: Virtual environment not found. Please run with venv setup.")
        return

    print(f"Exporting quotes from {csv_path} to {output_dir}...")
    
    count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row['id']
            quote_text = row['quote']
            
            # Use ID in filename to avoid duplicates
            out_path = os.path.join(output_dir, f"archived_{q_id}.png")
            
            if os.path.exists(out_path):
                # Skip if already exists
                continue
            
            # Generate image using USER theme for historical quotes
            subprocess.run([venv_python, "src/image_generator.py", quote_text, "USER", out_path, q_id])
            count += 1

    print(f"Done! Generated {count} new images.")

if __name__ == "__main__":
    export_all_quotes()
