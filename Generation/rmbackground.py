import os
import subprocess
from pathlib import Path

input_dir = Path("../data/2d_fishes")      
output_dir = Path("../data/2d_fishes")   
model = "u2net"                   

output_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for image_path in input_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        output_path = output_dir / (image_path.stem + ".png") 
        print(f"Processing: {image_path} → {output_path}")
        subprocess.run([
            "backgroundremover",
            "-i", str(image_path),
            "-o", str(output_path),
            "--model", model
        ])
