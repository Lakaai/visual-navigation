import cv2
import albumentations as A
import os
import random

# Define multiple augmentation pipelines with random parameter ranges for greater variety
augmentation_pipeline = A.Compose([
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
], additional_targets={'mask': 'mask'})

# Define input and output directories
input_dir = "/home/luke/Documents/Github/MCHA4400/lab12/data/train"
output_dir = "/home/luke/Documents/Github/MCHA4400/lab12/data/output"
output_annotated_dir = "/home/luke/Documents/Github/MCHA4400/lab12/data/output_annotated"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_annotated_dir, exist_ok=True)

# Number of augmented versions to create per image pair
num_augmented_versions = 5

# Loop through files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") and not filename.endswith("_annotated.png"):
        original_image_path = os.path.join(input_dir, filename)
        annotated_image_path = os.path.join(input_dir, filename.replace(".png", "_annotated.png"))

        if os.path.exists(original_image_path) and os.path.exists(annotated_image_path):
            original_image = cv2.imread(original_image_path)
            annotated_image = cv2.imread(annotated_image_path)

            for i in range(num_augmented_versions):
                augmented = augmentation_pipeline(image=original_image, mask=annotated_image)
                augmented_image = augmented['image']
                augmented_annotated = augmented['mask']

                # Save augmented images with unique names
                base_filename = filename.replace(".png", f"_augmented_{i}.png")
                cv2.imwrite(os.path.join(output_dir, base_filename), augmented_image)
                cv2.imwrite(os.path.join(output_annotated_dir, base_filename.replace(".png", "_annotated.png")), augmented_annotated)