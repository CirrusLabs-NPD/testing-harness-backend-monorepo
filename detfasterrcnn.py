from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from PIL import Image
from collections import defaultdict
import os
import pandas as pd
import torchvision.transforms as transforms

# Pre-trained model path
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Set model to evaluation mode
model.eval()

# Modify these arguments based on your data and training setup
conf_threshold = 0.5  # Adjust confidence threshold for detection

# Specify the directory containing your test images
test_dir = r'images'
labels_dir = r'labels'

# Define a transformation to apply to the image
transform = transforms.Compose([transforms.ToTensor()])


# Function to parse Faster R-CNN results into a DataFrame
def parse_results(results):
    data = defaultdict(list)

    for label, score, box in zip(results['labels'], results['scores'], results['boxes']):
        data['name'].append(label)
        data['confidence'].append(score)
        data['xmin'].append(box[0])
        data['ymin'].append(box[1])
        data['xmax'].append(box[2])
        data['ymax'].append(box[3])

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


# Function to perform prediction with Faster R-CNN
def predict_faster_rcnn(images, model, conf_threshold):
    with torch.no_grad():
        predictions = model(images)

    filtered_predictions = [{'labels': pred['labels'][pred['scores'] > conf_threshold].cpu().numpy(),
                             'scores': pred['scores'][pred['scores'] > conf_threshold].cpu().numpy(),
                             'boxes': pred['boxes'][pred['scores'] > conf_threshold].cpu().numpy()} for pred in
                            predictions]

    return filtered_predictions


# Function to calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou


# Function to compare predicted bounding boxes and class labels with ground truth annotations
def compare_prediction(predictions, ground_truth, iou_threshold=0.5):
    class_precision = defaultdict(float)
    class_recall = defaultdict(float)
    class_true_negatives = defaultdict(int)

    for gt_obj in ground_truth:
        gt_class = gt_obj["class"]
        gt_bbox = gt_obj["bbox"]
        class_tp_found = False

        for pred_obj in predictions:
            for label, score, box in zip(pred_obj['labels'], pred_obj['scores'], pred_obj['boxes']):
                pred_class = label
                pred_bbox = box
                iou = calculate_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold and pred_class == gt_class:
                    class_tp_found = True
                    break

        if class_tp_found:
            class_precision[gt_class] += 1
        else:
            class_recall[gt_class] += 1

    for gt_obj in ground_truth:
        gt_class = gt_obj["class"]
        class_true_negatives[gt_class] += len(predictions) - 1

    total_precision = sum(class_precision.values()) / len(class_precision) if len(class_precision) > 0 else 0
    total_recall = sum(class_recall.values()) / len(ground_truth) if len(ground_truth) > 0 else 0
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


# Loop through test images
for image_file in os.listdir(test_dir):
    if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".jpeg"):
        # Construct complete image path
        image_path = os.path.join(test_dir, image_file)

        # Load image and apply transformation
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        predictions = predict_faster_rcnn(image, model, conf_threshold)

        # Parse results into DataFrame
        prediction_df = parse_results(predictions[0])

        # Load ground truth annotations
        ground_truth = load_ground_truth_annotations(image_file, labels_dir)

        # Compare predictions with ground truth
        metrics = compare_prediction(predictions, ground_truth)

        print(f"Image: {image_file}")
        print(f"Metrics: {metrics}")





