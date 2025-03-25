# Bird or Not - Image Classification with FastAI

This project is an image classification model that distinguishes between images of birds and forests using FastAI and a pre-trained ResNet18 model.

## Features
- Downloads images from the web using DuckDuckGo image search.
- Organizes images into categories ("bird" and "forest").
- Preprocesses images by resizing and verifying their validity.
- Uses FastAI's DataBlock API to create DataLoaders.
- Trains a ResNet18 model to classify images.
- Makes predictions on new images.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```sh
  pip install fastai fastdownload duckduckgo-search pillow torchvision
  ```

## How to Run
1. **Ensure Internet Connection**
   - The script checks for internet connectivity before downloading images.

2. **Run the Script**
   ```sh
   python bird_or_not.py
   ```

3. **Model Training**
   - The script downloads images, preprocesses them, creates a dataset, and trains a ResNet18 model.

4. **Making Predictions**
   - The trained model predicts whether an image contains a bird.
   - The script selects an image from the "bird" category for testing.

## Folder Structure
```
project-folder/
│── bird_or_not/
│   ├── bird/     # Bird images
│   ├── forest/   # Forest images
│── script.py     # Main script
│── README.md     # Documentation
```

## Notes
- The dataset is automatically created and images are resized to a max size of 400 pixels.
- The validation dataset is 20% of the total images.
- If no valid images are found, ensure DuckDuckGo search API is working correctly.

## Troubleshooting
- **No images downloaded?**
  - Ensure your internet connection is stable.
  - Check if DuckDuckGo API is accessible.
- **Training dataset is empty?**
  - Verify that images were downloaded correctly.
  - Check the `failed` image count and manually inspect the image folder.

## License
This project is open-source and free to use.

