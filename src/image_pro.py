import subprocess
import os

def generate_pro_image(quote_text, output_path="quote.png", theme="GENERAL"):
    # Escape single quotes for shell
    safe_quote = quote_text.replace("'", "'\\''")
    safe_theme = theme.upper().replace("'", "'\\''")

    # ImageMagick Command
    # 1. Create a background (gradient + noise)
    # 2. Add the text with wrapping
    # 3. Add the theme header
    
    cmd = f"""magick -size 1080x1080 xc:black \
    -sparse-color Barycentric '0,0 #1a1a1a 1080,1080 #4b6cb7' \
    -fill white -alpha on -channel A -evaluate set 30% +channel \
    -draw "rectangle 50,50 1030,1030" \
    -fill white -alpha on -font DejaVu-Sans-Bold -pointsize 40 -gravity North -annotate +0+100 "PERSPECTIVE: {safe_theme}" \
    -fill white -alpha on -font DejaVu-Sans -pointsize 60 -gravity Center -size 900x caption:'{safe_quote}' \
    -fill white -alpha on -font DejaVu-Sans -pointsize 30 -gravity South -annotate +0+100 "SHITTY QUOTE ENGINE v2.0" \
    {output_path}"""

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Professional image saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating image: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1]
        theme = sys.argv[2] if len(sys.argv) > 2 else "GENERAL"
        generate_pro_image(text, theme=theme)
