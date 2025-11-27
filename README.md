# Object Detection System

Advanced object detection system using TensorFlow Hub and EfficientDet-D0. The project enables automatic detection and classification of objects in images with color recognition and result storage in a database.

## Features

- **Object Detection** - automatic detection and classification of objects in images
- **Color Recognition** - identification of dominant colors in detected objects
- **Multi-threaded Processing** - efficient processing of multiple images simultaneously
- **Database Storage** - SQLite with full multi-threading support
- **CSV Export** - results saved in CSV format for each image
- **Result Visualization** - automatic drawing of bounding boxes and labels on images
- **Linked List Implementation** - custom data structure for managing results
- **Unicode Support** - full support for special characters in file paths and names

## Technologies

- **Python 3.12**
- **TensorFlow 2.x** - machine learning framework
- **TensorFlow Hub** - EfficientDet-D0 model
- **OpenCV** - image processing
- **SQLAlchemy** - ORM for database management
- **Pandas** - data analysis and processing
- **NumPy** - numerical operations
- **Matplotlib** - result visualization

## Project Structure

```
Object_Detection/
├── data/
│   ├── images/                    # Input images
│   ├── labels.csv                 # Object class labels
│   └── detected_objects.db        # SQLite database
├── output/
│   ├── processed_images/          # Images with detected objects
│   └── processed_csv/             # CSV files with results
├── src/
│   ├── main.py                    # Main script
│   ├── detector.py                # Detection logic
│   ├── image_processor.py         # Single image processing
│   ├── image_display.py           # Result display
│   ├── database.py                # Database handling
│   ├── utils.py                   # Utility functions (colors)
│   ├── Queue.py                   # Queue implementation
│   └── Lista.py                   # Linked list implementation
└── model/                         # TensorFlow Hub model
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/misiakfilip/object-detection.git
cd object-detection
```

### 2. Create virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install tensorflow tensorflow-hub opencv-python pandas numpy matplotlib sqlalchemy
```

### 4. Download model

The EfficientDet-D0 model will be automatically downloaded from TensorFlow Hub on first run.

## Usage

### Basic execution

```bash
cd src
python main.py
```

### Configuration

In `main.py` you can adjust:

```python
num_threads = 4  # Number of threads for processing
accuracy = 0.5   # Detection confidence threshold (0.0 - 1.0)
```

### Adding images

1. Place your images in the `data/images/` folder
2. Supported formats: `.png`, `.jpg`, `.jpeg`
3. Run the script - results will appear in the `output/` folder

## Results

### Processed image
- Saved in `output/processed_images/`
- Contains bounding boxes around detected objects
- Labels with class name and color

### CSV file
- Saved in `output/processed_csv/`
- Columns: Class, Color, Scores, Box Coordinates

### Database
- Location: `data/detected_objects.db`
- Table: `detected_objects`
- Fields: id, box, Class, color, scores, box_coordinates

## Color Recognition

The system recognizes the following colors:

- Primary: Red, Blue, Green, Yellow, Orange
- Neutral: White, Black, Gray
- Additional: Pink, Purple, Brown, Cyan, and more

You can customize the algorithm in `utils.py`:
- `get_color_name()` - basic version (color dictionary)
- `get_color_name_advanced()` - advanced version (with thresholds)

## Multi-threading

The project uses multi-threading for parallel image processing:

- **Queue** - custom FIFO queue implementation
- **Threading** - Python threading module
- **Thread-safe database** - locks for safe writes

## Example Output

```
Detected objects:
{'Class': 'cup', 'Color': 'Blue', 'Scores': 0.84, 'Box Coordinates': (0.30, 0.58, 0.65, 0.85)}
{'Class': 'laptop', 'Color': 'Gray', 'Scores': 0.81, 'Box Coordinates': (0.27, 0.00, 0.98, 0.60)}
{'Class': 'cell phone', 'Color': 'Black', 'Scores': 0.70, 'Box Coordinates': (0.73, 0.66, 0.95, 0.95)}

Saved 3 objects to database
Successfully saved: image1_detect.jpg
Processing completed.
```

## License

This project is available under the MIT License. See the `LICENSE` file for details.

## Author

**Filip Misiak**

- GitHub: [@misiakfilip](https://github.com/misiakfilip)

**Adam Kowalczyk**

## Acknowledgments

- [TensorFlow Hub](https://tfhub.dev/) for object detection models
- [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) for model architecture
- OpenCV community for excellent image processing library

## Documentation

### Detected object classes

The project uses the COCO dataset with 90 object classes, including:
- People
- Vehicles (cars, bicycles, motorcycles)
- Animals (dogs, cats, horses)
- Everyday objects (phones, laptops, books)
- Food and beverages
- Furniture
- And many more...

### Customizing the model

To use a different model from TensorFlow Hub:

```python
def load_model():
    model_path = "https://tfhub.dev/tensorflow/efficientdet/d1/1"  # Example
    return hub.load(model_path)
```

### Database API

```python
from database import save_objects_data_to_database

objects_data = [
    {
        "Box": 1,
        "Class": "cup",
        "Color": "Blue",
        "Scores": 0.84,
        "Box Coordinates": "(0.30, 0.58, 0.65, 0.85)"
    }
]

save_objects_data_to_database(objects_data)
```

## Requirements

Install with:

```bash
pip install -r requirements.txt
```

## Performance

- Supports batch processing with configurable thread count
- Memory efficient with streaming processing
- Database operations optimized with connection pooling
