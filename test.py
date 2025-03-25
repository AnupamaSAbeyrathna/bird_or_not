from pathlib import Path
from fastai.vision.all import *

path = Path('bird_or_not')
image_files = get_image_files(path)

print(f"Total images found: {len(image_files)}")
print(image_files[:5])  # Show first 5 images
print([parent_label(f) for f in image_files[:-1]])

