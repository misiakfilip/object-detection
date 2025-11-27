import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image_unicode(filepath):
    """Read image with Unicode path support for Windows"""
    try:
        with open(filepath, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def display_saved_images(output_folder_images):
    """Display all processed images from the output folder"""

    # Check if folder exists
    if not os.path.exists(output_folder_images):
        print(f"Output folder does not exist: {output_folder_images}")
        return

    image_files = [
        f
        for f in os.listdir(output_folder_images)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print("No processed images found to display.")
        return

    print(f"Found {len(image_files)} processed images.")

    for image_file in image_files:
        image_path = os.path.join(output_folder_images, image_file)

        # Use Unicode-safe reading
        img = read_image_unicode(image_path)

        if img is None:
            print(f"Warning: Could not load {image_file}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(image_file)
        plt.axis("off")
        plt.show()