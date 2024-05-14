from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import json
import os 
import sys

def analyze_text_with_model(model_name, input_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    with open(input_file, 'r') as file:
        data = json.load(file)

    texts = [item['generated_text'] for item in data]
    original_scores = [item['score'] for item in data]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = softmax(logits, dim=1)
    model_scores = probabilities[:, 1]

    new_data = [
        {"generated_text": text, "original_score": original, "model_score": model_score.item()}
        for text, original, model_score in zip(texts, original_scores, model_scores)
    ]

    output_dir = "../outputs"  
    os.makedirs(output_dir, exist_ok=True)  

    output_file_name = os.path.basename(input_file).replace('.json', '_tested.json')
    output_file_path = os.path.join(output_dir, output_file_name)

    print("Saving output to:", output_dir)
    with open(output_file_path, 'w') as outfile:
        json.dump(new_data, outfile, indent=4)
    
    print("Main.py success")
    print(output_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py model_name input_file")
        sys.exit(1)

    model_name = sys.argv[1]
    input_file = sys.argv[2]
    analyze_text_with_model(model_name, input_file)
