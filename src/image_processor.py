import os

import cv2
import numpy as np


def read_image_unicode(filepath):
    try:
        with open(filepath, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def process_image(
        model, labels, image_name, input_folder, output_folder_images, output_folder_csv
):
    # Use os.path.join for proper path construction
    img_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(
        output_folder_images,
        f"{os.path.splitext(image_name)[0]}_detect.jpg"
    )
    output_path_csv = os.path.join(
        output_folder_csv,
        f"{os.path.splitext(image_name)[0]}_csv.csv"
    )

    # Use Unicode-safe reading method
    img = read_image_unicode(img_path)

    if img is not None:
        from detector import detect_objects
        detect_objects(
            model,
            img,
            0.5,
            output_path,
            output_path_csv,
            labels,
        )
    else:
        print(f"Error: Unable to read image {image_name}")