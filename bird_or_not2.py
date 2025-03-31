import multiprocessing
import socket
import time
from pathlib import Path
from fastdownload import download_url
from PIL import Image
from duckduckgo_search import DDGS
from fastai.vision.all import *
from torchvision.models import resnet18
from tqdm import tqdm

# Ensure proper multiprocessing support on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Check internet connectivity
    try:
        socket.setdefaulttimeout(1)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
    except socket.error:
        print("❌ ERROR: No internet connection. Please check your connection and try again.")
        exit(1)

    # Function to search for images
    def search_images(term, max_images=1):
        with DDGS() as ddgs:
            return [result["image"] for result in ddgs.images(term, max_results=max_images)]

    # Set up dataset directory
    searches = ['forest', 'bird']
    path = Path('dataset/bird_or_not')
    path.mkdir(exist_ok=True, parents=True)

    # Download images with progress bar
    for category in tqdm(searches, desc="Downloading images"):
        dest = path / category
        dest.mkdir(exist_ok=True, parents=True)
        urls = search_images(f'{category} photo', max_images=30)
        if not urls:
            print(f"⚠️ WARNING: No images found for '{category}'. Skipping...")
            continue
        print(f"Downloading {len(urls)} images for {category}...")
        download_images(dest, urls=urls)
        time.sleep(5)  # Prevent API request overload
        resize_images(dest, max_size=400)

    # Verify images
    failed = verify_images(get_image_files(path))
    if failed:
        print(f"⚠️ WARNING: {len(failed)} images failed verification and will be removed.")
        failed.map(Path.unlink)
    else:
        print("✅ All images verified successfully!")

    # Count valid images
    valid_images = get_image_files(path)
    print(f"📸 Total valid images found: {len(valid_images)}")

    # Check if dataset is empty
    if len(valid_images) == 0:
        print("❌ ERROR: No valid images found. Ensure images are downloaded correctly.")
        exit(1)

    # Create DataLoaders
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    )

    dls = dblock.dataloaders(path, bs=8)

    # Debugging: Check dataset sizes
    print(f"📊 Training set size: {len(dls.train_ds)}")
    print(f"📊 Validation set size: {len(dls.valid_ds)}")

    # Ensure DataLoader is not empty
    if len(dls.train_ds) == 0:
        print("❌ ERROR: Training dataset is empty! Adjust validation split or check images.")
        exit(1)

    # Show a sample batch
    dls.show_batch(max_n=6)

    # Train model using vision_learner
    print("🚀 Training model...")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    print("✅ Model training complete!")

    # Make a prediction on an image
    valid_bird_images = get_image_files(path/'bird')
    if valid_bird_images:
        print("🔍 Making a prediction...")
        is_bird, _, probs = learn.predict(PILImage.create(valid_bird_images[0]))
        print(f"🔹 This is a: {is_bird}")
        print(f"📈 Probability it's a {is_bird}: {probs.max():.4f}")
    else:
        print("⚠️ No bird images found for prediction.")

