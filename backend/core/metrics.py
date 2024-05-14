from nltk.translate import meteor_score
import nltk
import evaluate
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from django.conf import settings
import pandas as pd
import nltk
import numpy as np
import os
import base64
import io
from sklearn.metrics import accuracy_score

# Calculate METEOR scores
def calculate_meteor(tokenized_df, tokenized_predicted):
    try:
        meteor_scores = []
        for reference, hypothesis in zip(tokenized_df['fr_tokenized'], tokenized_predicted):
            score = meteor_score.meteor_score([reference], hypothesis)
            meteor_scores.append(score)
        return meteor_scores
    except Exception as identifier:
        print(f"Got the following error (METEOR): {identifier}")
        return
    
# Calculate BLEU score
def calculate_bleu(predictions, references):
    try:
        bleu = evaluate.load("bleu")
        # print("Predictions:")
        # pprint.pprint(predictions[:5])
        # print("\nReferences:")
        # pprint.pprint(references[:5])
        bleu_score = bleu.compute(predictions=predictions, references=references)
        return bleu_score
    except Exception as identifier:
        print(f"Got the following error (BLEU): {identifier}")
        return
    
# Calculate TER score
def calculate_ter(predictions, references):
    try:
        ter = evaluate.load("ter")
        ter_score = ter.compute(predictions=predictions, references=references, ignore_punct=True, case_sensitive=False)
        return ter_score
    except Exception as identifier:
            print(f"Got the following error (TER): {identifier}")
            return

def calculate_accuracy(predictions, references):
    try:
        accuracy = accuracy_score(references, predictions)
        # print(f"Accuracy: {accuracy}")
        return accuracy
    except Exception as identifier:
        print(f"Got the following error (ACCURACY): {identifier}")
        return

# Test the given translations against all metrics
def test_metrics(clean_df, tokenized_df, predicted_df):
    # print(f"PREDICTED_DF: {predicted_df}")
    predicted_tokenized = predicted_df['fr'].apply(nltk.word_tokenize)
    # print(f"PREDICTED_TOKENIZED DF: {predicted_tokenized}")
    predictions = predicted_df['fr'].tolist()
    references = [value for value in clean_df['fr']]
    clean_df['predicted'] = predicted_df['fr']
    # print(f"TOKENIZED_DF: {tokenized_df}")
    # print(f"PREDICTED_TOKENIZED: {predicted_tokenized}")
    meteor_scores = calculate_meteor(tokenized_df, predicted_tokenized)
    # print(f"CLEAN_DF: {clean_df}")
    bleu_score = calculate_bleu(predictions, references)
    ter_score = calculate_ter(predictions, references)
    accuracy_score = calculate_accuracy(predictions, references)
    
    return bleu_score, ter_score, meteor_scores, accuracy_score

# Save the metric results to file
def save_metrics(name, content):
    if isinstance(content, pd.DataFrame):
        content.to_csv(f"{name}.txt", index=False)
    else:
        f = open(f"{name}.txt", "w")
        f.write(str(content))
        f.close()
    
# Rounds down scores to 3 decimal places
def clean_scores(scores):
    if isinstance(scores, dict):
        if 'bleu' in scores.keys():
            scores["bleu"] = round(scores["bleu"], 3)
        else:
            scores["score"] = round(scores["score"], 3)
        return scores
    else:
        cleaned_scores = []
        for value in scores:
            cleaned_scores.append(round(value, 3))
        return cleaned_scores

# Visualize the metric results as graphs/charts
def visualize_bleu(t5, f200, hel):
    plt.figure(1)
    graph_df = {'T5': t5['bleu'], 'F200': f200['bleu'], 'HEL': hel['bleu']}
    ax = sns.barplot(data=graph_df, color='blue')
    ax.bar_label(ax.containers[0])
    plt.title("BLEU Score Distribution")
    plt.xlabel("Models")
    plt.ylabel("BLEU Scores")
    return

# Visualize the metric results as graphs/charts
def visualize_ter(t5, f200, hel):
    plt.figure(2)
    graph_df = {'T5': t5['score'], 'F200': f200['score'], 'HEL': hel['score']}
    ax = sns.barplot(data=graph_df, color='orange')
    ax.bar_label(ax.containers[0])
    plt.title("TER Score Distribution")
    plt.xlabel("Models")
    plt.ylabel("TER Scores")
    return

def visualize_meteor(t5, f200, hel):
    plt.figure(3)
    data = list(zip(t5, f200, hel))
    df = pd.DataFrame(data, columns=['T5', 'F200', 'HEL'])
    graph_df = df.melt(var_name='Model', value_name='Meteor Score')
    graph_df = graph_df.reset_index(drop=True)
    print(df)
    print(graph_df)
    sns.histplot(graph_df, x="Meteor Score", hue="Model", element="poly")
    plt.title("METEOR Score Distribution")
    plt.xlabel("METEOR Scores")
    plt.ylabel("Count")

def print_meteor(name, scores, datasets, models):
    plt.figure()
    dfs = {}
    for dataset in datasets:
        df = pd.DataFrame({model[0]: scores[dataset][model] for model in models})
        graph_df = df.melt(var_name='Model', value_name='Meteor Score')
        graph_df['Dataset'] = dataset
        dfs[dataset] = graph_df

    merged_df = pd.concat(dfs.values())
    sns.histplot(data=merged_df, x="Meteor Score", hue="Model", col="Dataset", multiple="dodge", element="poly")
    plt.suptitle(f"{name} Score Distribution")
    plt.xlabel("METEOR Scores")
    plt.ylabel("Count")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Convert the plot to a base64-encoded string
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    media_path = os.path.join(settings.MEDIA_ROOT, f"{name}_distribution.png")
    with open(media_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_image))

    plt.close()
    return media_path


def print_bleu_ter(name, colors, scores, datasets, models):
    plt.figure(figsize=(10, 6))
    width = 0.35  # Width of the bars

    # Initialize the bottom values for stacking bars
    bottom_values = np.zeros(len(models))

    # Initialize the figure and axis outside the loop
    fig, ax = plt.subplots()

    legend_patches = []  # List to store legend patches

    for dataset in datasets:
        for x, model in enumerate(models):
            if name.lower() == 'bleu':
                score = scores[dataset][x]['bleu']
            elif name.lower() == 'ter':
                score = scores[dataset][x]['score']
            else:
                score = scores[dataset][x]
            
            ax.bar(model, score, width, bottom=bottom_values[x], color=colors[dataset])
            bottom_values[x] += score

        # Create a legend entry for the dataset
        legend_patches.append(Patch(color=colors[dataset], label=dataset))

    # Add legend with custom legend patches
    ax.legend(handles=legend_patches)
    ax.set_title(f"{name} Score Distribution")
    ax.set_xlabel("Models")
    ax.set_ylabel(f"{name} Scores")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Adjust layout
    fig.tight_layout()

    # Save the plot only once after the loop
    media_path = os.path.join(settings.MEDIA_ROOT, f"{name}_distribution.png")
    plt.savefig(media_path)
    plt.close(fig)

    return media_path