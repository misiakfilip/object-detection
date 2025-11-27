import csv
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from Lista import LinkedList
from database import save_objects_data_to_database
from utils import get_color_name

tf.config.optimizer.set_experimental_options({"compile": True})


def load_model():
    model_path = "model"
    return hub.load(model_path)


def detect_objects(
        model, img, accuracy, output_path, output_path_csv, labels, target_class=None
):
    linked_list = LinkedList()
    objects_data = []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    result = model(rgb_tensor)

    boxes = result["detection_boxes"].numpy()
    classes = result["detection_classes"].numpy().astype("int")[0]
    pred_labels = [labels[i] for i in classes]
    scores = result["detection_scores"].numpy()[0]

    for score, (ymin, xmin, ymax, xmax), label in zip(scores, boxes[0], pred_labels):
        if score >= accuracy:
            x, y, w, h = (
                int(xmin * img.shape[1]),
                int(ymin * img.shape[0]),
                int((xmax - xmin) * img.shape[1]),
                int((ymax - ymin) * img.shape[0]),
            )

            object_region = img[y : y + h, x : x + w]
            average_color = np.mean(object_region, axis=(0, 1)).astype(int)
            color_name = get_color_name(average_color)

            linked_list.append(
                {
                    "Class": label,
                    "Color": color_name,
                    "Scores": score,
                    "Box Coordinates": (ymin, xmin, ymax, xmax),
                }
            )
            objects_data.append(
                {
                    "Box": len(objects_data) + 1,
                    "Class": label,
                    "Color": color_name,
                    "Scores": score,
                    "Box Coordinates": f"({ymin}, {xmin}, {ymax}, {xmax})",
                }
            )

            color = (0, 255, 0)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img, f"{label} ({color_name})", (x, y - 10), font, 0.5, (0, 255, 0), 2
            )

    print("Wykryte obiekty:")
    linked_list.display()

    save_objects_data_to_database(objects_data)

    objects_data = linked_list.to_list()
    csv_columns = ["Class", "Color", "Scores", "Box Coordinates"]

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Unicode-safe image writing
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        if is_success:
            im_buf_arr.tofile(output_path)
            print(f"Successfully saved: {os.path.basename(output_path)}")
        else:
            print(f"Warning: Failed to encode image for {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")