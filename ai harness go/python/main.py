from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import json
import os  # Import the os module
import sys

def analyze_text_with_model(model_name, input_file):
    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load data from the user-specified input file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Extract texts and original scores
    texts = [item['generated_text'] for item in data]
    original_scores = [item['score'] for item in data]

    # Tokenize texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Compute probabilities and extract model scores
    probabilities = softmax(logits, dim=1)
    model_scores = probabilities[:, 1]

    # Prepare new data structure with results
    new_data = [
        {"generated_text": text, "original_score": original, "model_score": model_score.item()}
        for text, original, model_score in zip(texts, original_scores, model_scores)
    ]

    # Ensure the outputs directory exists
    output_dir = "../outputs"  # Define the outputs directory relative path
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the output file based on the input file, to be saved in the outputs directory
    output_file_name = os.path.basename(input_file).replace('.json', '_tested.json')
    output_file_path = os.path.join(output_dir, output_file_name)

    print("Saving output to:", output_dir)
    # Save results to the dynamically generated output file in the outputs directory
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
    
    # Now call your function with the command-line arguments
    analyze_text_with_model(model_name, input_file)
