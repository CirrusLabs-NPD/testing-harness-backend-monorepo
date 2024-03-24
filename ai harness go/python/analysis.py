# analysis.py
import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import os

def analyze_data(file_name):
    print(file_name)
    input_filepath = os.path.join("..", "outputs", file_name)
    print(input_filepath)
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    y_true = np.array([item['original_score'] for item in data])
    y_scores = np.array([item['model_score'] for item in data])

    threshold = 0.5
    y_pred = np.where(y_scores >= threshold, 1, 0)

    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    output_dir = "../outputs"
    results_file_name = input_filepath.replace('.json', '_results.json')
    results_filepath = os.path.join(output_dir, results_file_name)
    with open(results_filepath, 'w') as results_file:
        json.dump(results, results_file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py model_name input_file")
        sys.exit(1)

    file_name = sys.argv[1]
    analyze_data(file_name)