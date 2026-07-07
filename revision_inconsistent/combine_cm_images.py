import os
import glob
from PIL import Image, ImageDraw, ImageFont

def combine_images():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind Training Resource/revision_inconsistent/cm_images_v2"
    out_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind Training Resource/revision_inconsistent"
    
    tables = ["T1", "T2", "T3", "T4"]
    
    try:
        font = ImageFont.truetype("Arial", 40)
    except:
        font = ImageFont.load_default()
        
    for table in tables:
        files = glob.glob(os.path.join(base_dir, f"{table}_*.png"))
        if not files:
            continue
            
        images = [Image.open(f) for f in sorted(files)]
        
        w, h = images[0].size
        padding = 50
        title_space = 100
        
        # Calculate grid size (max 3 columns)
        cols = min(3, len(images))
        rows = (len(images) + cols - 1) // cols
        
        total_width = w * cols + padding * (cols + 1)
        total_height = h * rows + padding * (rows + 1) + title_space
        
        combined = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(combined)
        
        title = f"Combined Confusion Matrices - {table}"
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((total_width - tw) / 2, padding), title, fill='black', font=font)
        
        y = padding + title_space
        x = padding
        for i, img in enumerate(images):
            combined.paste(img, (x, y))
            x += w + padding
            if (i + 1) % cols == 0:
                x = padding
                y += h + padding
                
        out_path = os.path.join(out_dir, f"Combined_{table}_CMs.png")
        combined.save(out_path)
        print(f"Saved combined image to {out_path}")

if __name__ == "__main__":
    combine_images()
