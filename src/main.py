import os
import warnings

# Wycisz warningi TensorFlow i inne
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Wyłącz oneDNN warnings

# Wycisz Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Wycisz TensorFlow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import threading

import cv2
import pandas as pd

from Queue import Queue
from detector import load_model
from image_display import display_saved_images
from image_processor import process_image


def worker(
    custom_queue, model, labels, input_folder, output_folder_images, output_folder_csv
):
    while not custom_queue.is_empty():
        image_name = custom_queue.dequeue()
        try:
            process_image(
                model,
                labels,
                image_name,
                input_folder,
                output_folder_images,
                output_folder_csv,
            )
        finally:
            pass


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_folder = os.path.join(base_dir, "data", "images")
    labels_files = os.path.join(base_dir, "data", "labels.csv")
    output_folder_images = os.path.join(base_dir, "output", "processed_images")
    output_folder_csv = os.path.join(base_dir, "output", "processed_csv")

    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_csv, exist_ok=True)

    image_names = [
        file
        for file in os.listdir(input_folder)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    model = load_model()
    labels = pd.read_csv(labels_files, sep=";", index_col="ID")["OBJECT (2017 REL.)"]

    from database import init_db
    init_db()

    custom_queue = Queue()
    for image_name in image_names:
        custom_queue.enqueue(image_name)

    num_threads = 4  # Adjust the number of threads as needed
    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(
            target=worker,
            args=(
                custom_queue,
                model,
                labels,
                input_folder,
                output_folder_images,
                output_folder_csv,
            ),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("Processing completed.")
    display_saved_images(output_folder_images)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
