from ultralytics import YOLO
from collections import defaultdict
import os
import pandas as pd

# Pre-trained model path
model = YOLO('yolov8n.pt')

# Modify these arguments based on your data and training setup
imgsz = 640  # Adjust input image size as needed
conf = 0.5  # Adjust confidence threshold for detection

# Specify the directory containing your test images
test_dir = r'images'
labels_dir = r'labels'

# Function to parse YOLO results into a DataFrame
def parse_results(results):
    data = defaultdict(list)

    # Check if the results contain the 'boxes' attribute
    if 'boxes' in results:
        # Extract relevant information from the 'boxes' attribute
        for label, conf, box in zip(results.names, results.boxes.get_field('scores'), results.boxes.tolist()):
            data['name'].append(label)
            data['confidence'].append(conf)
            data['xmin'].append(box[0])
            data['ymin'].append(box[1])
            data['xmax'].append(box[2])
            data['ymax'].append(box[3])
    else:
        # Handle cases where 'boxes' attribute is not present
        print("Error: 'boxes' attribute not found in results.")

    return pd.DataFrame(data)


# Function to load ground truth annotations
def load_ground_truth_annotations(image_file, labels_dir):
    label_file = os.path.join(labels_dir, image_file.replace(".jpg", ".txt"))
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            class_index, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = max(0, (x_center - width / 2))
            y_min = max(0, (y_center - height / 2))
            x_max = min(1, (x_center + width / 2))
            y_max = min(1, (y_center + height / 2))
            annotation = {
                "class": int(class_index),
                "bbox": (x_min, y_min, x_max, y_max)
            }
            annotations.append(annotation)
        return annotations
    else:
        return []



def compare_prediction(prediction, ground_truth, iou_threshold=0.5):
    """
    Compares predicted bounding boxes and class labels with ground truth annotations.
    Calculates precision, recall, and true negatives for each class.

    Args:
        prediction: A pandas DataFrame containing predicted bounding boxes and class labels.
        ground_truth: A list of dictionaries representing ground truth annotations for each object.
        iou_threshold: Threshold for considering a prediction as correct based on IoU.

    Returns:
        A dictionary containing precision, recall, and true negatives for each class.
    """

    # Initialize dictionaries to store metrics per class
    class_precision = defaultdict(float)
    class_recall = defaultdict(float)
    class_true_negatives = defaultdict(int)

    # Loop through each ground truth object
    for gt_obj in ground_truth:
        gt_class = gt_obj["class"]
        gt_bbox = gt_obj["bbox"]

        # Flag to indicate if a true positive was found for this class
        class_tp_found = False

        # Loop through each predicted object
        for _, p_obj in prediction.iterrows():
            p_class = p_obj['name']
            p_bbox = (p_obj['xmin'], p_obj['ymin'], p_obj['xmax'], p_obj['ymax'])

            # Calculate IoU between ground truth and prediction
            iou = calculate_iou(gt_bbox, p_bbox)

            # Check if the prediction is a true positive for the current class
            if iou >= iou_threshold and p_class == gt_class:
                class_tp_found = True
                break

        # Update metrics based on the presence of a true positive for the class
        if class_tp_found:
            class_precision[gt_class] += 1
        else:
            class_recall[gt_class] += 1

    # Calculate true negatives for each class (assuming all objects are present in ground truth)
    for gt_obj in ground_truth:
        gt_class = gt_obj["class"]
        class_true_negatives[gt_class] += len(prediction) - 1  # Subtract 1 to exclude the current prediction

    # Calculate overall precision
    total_precision = sum(class_precision.values()) / len(class_precision) if len(class_precision) > 0 else 0

    # Calculate overall recall (assuming all ground truth objects are detected)
    total_recall = sum(class_recall.values()) / len(ground_truth) if len(ground_truth) > 0 else 0

    # Calculate total true negatives for all classes
    total_true_negatives = sum(class_true_negatives.values())

    return {
        "precision": total_precision,
        "recall": total_recall,
        "true_negatives": total_true_negatives,
        "class_wise_metrics": {
            cls: {
                "precision": class_precision[cls],
                "recall": class_recall[cls],
                "true_negatives": class_true_negatives[cls]
            }
            for cls in class_precision
        }
    }


def calculate_iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: A tuple representing a bounding box (xmin, ymin, xmax, ymax).
        bbox2: A tuple representing another bounding box (xmin, ymin, xmax, ymax).

    Returns:
        The Intersection over Union (IoU) value between the two bounding boxes.
    """

    # Extract coordinates from bounding boxes
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate the area of overlap
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate and return the IoU
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou




for image_file in os.listdir(test_dir):
    if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".jpeg"):
        # Construct complete image path
        image_path = os.path.join(test_dir, image_file)

        # Perform prediction and save results
        results = model.predict(image_path, save=True, imgsz=imgsz, conf=conf)


        # Parse YOLO results into DataFrame
        prediction = parse_results(results)
        # Print the type and structure of the results object

        # Load ground truth annotations
        ground_truth = load_ground_truth_annotations(image_file, labels_dir)

        # Compare predictions with ground truth
        metrics = compare_prediction(prediction, ground_truth)
        #accuracy,precision, f1-score.


        print(f"Image: {image_file}")
        print(f"Metrics: {metrics}")