from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
import os

def generate_image(text, theme="GENERAL", output_path="quote.png", bg_image_path="clound.jpg", seed=None):
    # Image size
    width, height = 1080, 1080
    
    # Load background image or fallback
    if os.path.exists(bg_image_path):
        bg = Image.open(bg_image_path)
        bg_ratio = bg.width / bg.height
        target_ratio = width / height
        if bg_ratio > target_ratio:
            new_width = int(target_ratio * bg.height)
            left = (bg.width - new_width) / 2
            bg = bg.crop((left, 0, left + new_width, bg.height))
        else:
            new_height = int(bg.width / target_ratio)
            top = (bg.height - new_height) / 2
            bg = bg.crop((0, top, bg.width, top + new_height))
        base = bg.resize((width, height), Image.Resampling.LANCZOS)
        
        # Add a dark overlay
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 110))
        base = base.convert('RGBA')
        base = Image.alpha_composite(base, overlay).convert('RGB')
    else:
        base = Image.new('RGB', (width, height), (30, 30, 30))

    draw = ImageDraw.Draw(base)
    
    # Fonts
    font_path_serif_italic = "/nix/store/6kc9n2dxvnd9dsqb4ymsi6hhcj0r5fmy-dejavu-fonts-2.37/share/fonts/truetype/DejaVuSerif-Italic.ttf"
    font_path_light = "/nix/store/6kc9n2dxvnd9dsqb4ymsi6hhcj0r5fmy-dejavu-fonts-2.37/share/fonts/truetype/DejaVuSans-ExtraLight.ttf"
    
    if not os.path.exists(font_path_serif_italic):
        font_main = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    else:
        font_main = ImageFont.truetype(font_path_serif_italic, 65)
        font_tiny = ImageFont.truetype(font_path_light, 18) # Ultra small

    # Wrap and Draw Main Text
    wrapped_text = textwrap.fill(text, width=28)
    lines = wrapped_text.split('\n')
    line_spacing = 20
    total_text_height = sum(draw.textbbox((0, 0), line, font=font_main)[3] for line in lines) + (len(lines) - 1) * line_spacing
    
    current_y = (height - total_text_height) / 2
    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font_main)
        line_width = line_bbox[2] - line_bbox[0]
        draw.text(((width - line_width) / 2 + 2, current_y + 2), line, font=font_main, fill=(0, 0, 0, 150))
        draw.text(((width - line_width) / 2, current_y), line, font=font_main, fill=(255, 255, 255))
        current_y += line_bbox[3] - line_bbox[1] + line_spacing

    # Combined Ultra-Small Info Row
    footer_y = height - 60
    display_seed = (seed[:8] + "...") if seed and len(seed) > 12 else (seed or "N/A")
    info_text = f"THEME: {theme.lower()}  |  SEED: {display_seed}  |  ENGINE: shitty v2.8"
    
    # Draw Info Row (Centered and very subtle)
    i_bbox = draw.textbbox((0, 0), info_text, font=font_tiny)
    i_width = i_bbox[2] - i_bbox[0]
    draw.text(((width - i_width) / 2, footer_y), info_text, font=font_tiny, fill=(255, 255, 255, 70)) # High transparency

    base.save(output_path)
    print(f"Ultra-minimalist image saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1]
        theme = sys.argv[2] if len(sys.argv) > 2 else "GENERAL"
        output = sys.argv[3] if len(sys.argv) > 3 else "quote.png"
        seed = sys.argv[4] if len(sys.argv) > 4 else None
        generate_image(text, theme=theme, output_path=output, seed=seed)
