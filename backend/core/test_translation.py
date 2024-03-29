from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import nltk
import evaluate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from django.conf import settings
import pandas as pd
import nltk
from datasets import load_dataset
import os
import base64
import io

matplotlib.use('agg')

# Define paths to data and result folders
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

# nltk.download('wordnet')

# Preprocess the given df
def clean(clean_df):
    # Lowercase
    clean_df['en'] = clean_df['en'].str.lower()
    clean_df['fr'] = clean_df['fr'].str.lower()

    # Filter by length
    clean_df = clean_df[(clean_df['en'].str.len() <= 50) & (clean_df['fr'].str.len() <= 50)]
    clean_df.reset_index(drop=True, inplace=True)

    tokenized_df = pd.DataFrame()

    # Tokenize sentences
    tokenized_df['en_tokenized'] = clean_df['en'].apply(nltk.word_tokenize)
    tokenized_df['fr_tokenized'] = clean_df['fr'].apply(nltk.word_tokenize)


    # print(clean_df.head())
    return clean_df, tokenized_df

# Get clean and tokenized default dataset
df = pd.read_csv(os.path.join(DATA_FOLDER, 'en-fr.csv') , nrows=2000)
df.to_csv(os.path.join(DATA_FOLDER, 'base_df.csv') , index=False)
clean_df, tokenized_df = clean(df)
# print(f"TOKENIZED DF: {tokenized_df}")
clean_df.to_csv(os.path.join(DATA_FOLDER, 'clean_df.csv') , index=False)

# Load and arrange medical dataset properly
dataset = load_dataset('qanastek/WMT-16-PubMed', split=['train[0:5000]'], trust_remote_code=True)
temp = pd.DataFrame(dataset)
temp = temp.transpose()

en_column = []
fr_column = []
# print(temp)
for sentence in temp[0]:
        en = sentence['translation']['en']
        fr = sentence['translation']['fr']
        
        en_column.append(en)
        fr_column.append(fr)

medf = pd.DataFrame()
medf['en'] = en_column
medf['fr'] = fr_column

# Get clean and tokenized medical dataset
clean_medf, tokenized_medf = clean(medf)
clean_medf.to_csv(os.path.join(DATA_FOLDER, 'clean_medf.csv'), index=False)

def fix_df(df):
    df = df.iloc[1:]
    df.reset_index(drop=True, inplace=True)
    return df

# Get the predictions
predicted_t5_df = pd.read_csv(os.path.join(DATA_FOLDER, 'predicted_t5_df.csv'), header=None, names=['fr'])
predicted_t5_df = fix_df(predicted_t5_df)

predicted_f200_df = pd.read_csv(os.path.join(DATA_FOLDER, 'predicted_f200_df.csv'), header=None, names=['fr'])
predicted_f200_df = fix_df(predicted_f200_df)

predicted_hel_df = pd.read_csv(os.path.join(DATA_FOLDER, 'predicted_hel_df.csv'), header=None, names=['fr'])
predicted_hel_df = fix_df(predicted_hel_df)

predicted_t5_medf = pd.read_csv(os.path.join(DATA_FOLDER, 'predicted_t5_medf.csv'), header=None, names=['fr'])
predicted_t5_medf = fix_df(predicted_t5_medf)

predicted_f200_medf = pd.read_csv(os.path.join(DATA_FOLDER, 'predicted_f200_medf.csv'), header=None, names=['fr'])
predicted_f200_medf = fix_df(predicted_f200_medf)

predicted_hel_medf = pd.read_csv(os.path.join(DATA_FOLDER, 'predicted_hel_medf.csv'), header=None, names=['fr'])
predicted_hel_medf = fix_df(predicted_hel_medf)

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
    
    return clean_df, bleu_score, ter_score, meteor_scores

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

def print_meteor(name, scores, models):
    plt.figure()
    df = pd.DataFrame({model[0]: score for model, score in zip(models, scores)})
    graph_df = df.melt(var_name='Model', value_name='Meteor Score')
    graph_df = graph_df.reset_index(drop=True)
    sns.histplot(graph_df, x="Meteor Score", hue="Model", element="poly")
    plt.title(f"{name} Score Distribution")
    plt.xlabel("METEOR Scores")
    plt.ylabel("Count")
    # plt.savefig(f"{name}_distribution.png")
    
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

def print_bleu_ter(name, color, scores, models):
    plt.figure()
    graph_df = {}
    for x in range(len(models)):
        if name.lower() == 'bleu':
            graph_df[models[x]] = scores[x]['bleu']
        else:
            graph_df[models[x]] = scores[x]['score']
    ax = sns.barplot(data=graph_df, color=color)
    ax.bar_label(ax.containers[0])
    plt.title(f"{name} Score Distribution")
    plt.xlabel("Models")
    plt.ylabel(f"{name} Scores")
    # plt.savefig(f"{name}_distribution.png")
    
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

# Test the models on the default dataset
def test_df(selected_models):
    # Initialize lists to store results for each model
    final_results = []
    bleu_results = []
    ter_results = []
    meteor_results = []
    
    # Check if the model is present and perform test metrics if it is
    if 'Google T5' in selected_models:
        final_t5_df, bleu_t5_df, ter_t5_df, meteor_t5_df = test_metrics(clean_df, tokenized_df, predicted_t5_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/t5/T5_meteor'), meteor_t5_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/t5/T5_bleu'), bleu_t5_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/t5/T5_ter'), ter_t5_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/t5/T5_translations'), final_t5_df)
        
        meteor_t5_df = clean_scores(meteor_t5_df)
        bleu_t5_df = clean_scores(bleu_t5_df)
        ter_t5_df = clean_scores(ter_t5_df)
        
        final_results.append(final_t5_df)
        meteor_results.append(meteor_t5_df)
        bleu_results.append(bleu_t5_df)
        ter_results.append(ter_t5_df)
    
    if 'Facebook NLLB' in selected_models:
        final_f200_df, bleu_f200_df, ter_f200_df, meteor_f200_df = test_metrics(clean_df, tokenized_df, predicted_f200_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/f200/F200_meteor'), meteor_f200_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/f200/F200_bleu'), bleu_f200_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/f200/F200_ter'), ter_f200_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/f200/F200_translations'), final_f200_df)
        
        meteor_f200_df = clean_scores(meteor_f200_df)
        bleu_f200_df = clean_scores(bleu_f200_df)
        ter_f200_df = clean_scores(ter_f200_df)
        
        final_results.append(final_f200_df)
        meteor_results.append(meteor_f200_df)
        bleu_results.append(bleu_f200_df)
        ter_results.append(ter_f200_df)
    
    if 'Helsinki Opus' in selected_models:
        final_hel_df, bleu_hel_df, ter_hel_df, meteor_hel_df = test_metrics(clean_df, tokenized_df, predicted_hel_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/hel/HEL_meteor'), meteor_hel_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/hel/HEL_bleu'), bleu_hel_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/hel/HEL_ter'), ter_hel_df)
        save_metrics(os.path.join(RESULTS_FOLDER, 'df/hel/HEL_translations'), final_hel_df)
        
        meteor_hel_df = clean_scores(meteor_hel_df)
        bleu_hel_df = clean_scores(bleu_hel_df)
        ter_hel_df = clean_scores(ter_hel_df)
        
        final_results.append(final_hel_df)
        meteor_results.append(meteor_hel_df)
        bleu_results.append(bleu_hel_df)
        ter_results.append(ter_hel_df)
    
    bleu_image = print_bleu_ter("bleu", "blue", bleu_results, selected_models)
    ter_image = print_bleu_ter("ter", "orange", ter_results, selected_models)
    meteor_image = print_meteor("meteor", meteor_results, selected_models)
    
    return final_results, bleu_image, ter_image, meteor_image

# Test the models on the medical dataset
def test_medf(selected_models):
    # Initialize lists to store results for each model
    final_results = []
    bleu_results = []
    ter_results = []
    meteor_results = []
    
    # Check if the model is present and perform test metrics if it is
    if 'Google T5' in selected_models:
        final_t5_medf, bleu_t5_medf, ter_t5_medf, meteor_t5_medf = test_metrics(clean_medf, tokenized_medf, predicted_t5_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/t5/T5_meteor'), meteor_t5_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/t5/T5_bleu'), bleu_t5_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/t5/T5_ter'), ter_t5_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/t5/T5_translations'), final_t5_medf)
        
        meteor_t5_medf = clean_scores(meteor_t5_medf)
        bleu_t5_medf = clean_scores(bleu_t5_medf)
        ter_t5_medf = clean_scores(ter_t5_medf)
        
        final_results.append(final_t5_medf)
        meteor_results.append(meteor_t5_medf)
        bleu_results.append(bleu_t5_medf)
        ter_results.append(ter_t5_medf)
    
    if 'Facebook NLLB' in selected_models:
        final_f200_medf, bleu_f200_medf, ter_f200_medf, meteor_f200_medf = test_metrics(clean_medf, tokenized_medf, predicted_f200_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/f200/F200_meteor'), meteor_f200_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/f200/F200_bleu'), bleu_f200_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/f200/F200_ter'), ter_f200_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/f200/F200_translations'), final_f200_medf)
        
        meteor_f200_medf = clean_scores(meteor_f200_medf)
        bleu_f200_medf = clean_scores(bleu_f200_medf)
        ter_f200_medf = clean_scores(ter_f200_medf)
        
        final_results.append(final_f200_medf)
        meteor_results.append(meteor_f200_medf)
        bleu_results.append(bleu_f200_medf)
        ter_results.append(ter_f200_medf)
    
    if 'Helsinki Opus' in selected_models:
        final_hel_medf, bleu_hel_medf, ter_hel_medf, meteor_hel_medf = test_metrics(clean_medf, tokenized_medf, predicted_hel_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/hel/HEL_meteor'), meteor_hel_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/hel/HEL_bleu'), bleu_hel_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/hel/HEL_ter'), ter_hel_medf)
        save_metrics(os.path.join(RESULTS_FOLDER, 'medf/hel/HEL_translations'), final_hel_medf)
        
        meteor_hel_medf = clean_scores(meteor_hel_medf)
        bleu_hel_medf = clean_scores(bleu_hel_medf)
        ter_hel_medf = clean_scores(ter_hel_medf)
        
        final_results.append(final_hel_medf)
        meteor_results.append(meteor_hel_medf)
        bleu_results.append(bleu_hel_medf)
        ter_results.append(ter_hel_medf)
    
    bleu_image = print_bleu_ter("bleu", "blue", bleu_results, selected_models)
    ter_image = print_bleu_ter("ter", "orange", ter_results, selected_models)
    meteor_image = print_meteor("meteor", meteor_results, selected_models)
    
    return final_results, bleu_results, ter_results, meteor_results

def test_user(selected_dataset, selected_models):
    if selected_dataset == 'Standard':
        return test_df(selected_models)
    elif selected_dataset == 'Medical':
        return test_medf(selected_models)

# visualize_bleu(bleu_t5_df, bleu_f200_df, bleu_hel_df)
# visualize_ter(ter_t5_df, ter_f200_df, ter_hel_df)
# visualize_meteor(meteor_t5_df, meteor_f200_df, meteor_hel_df)
# plt.show()
# visualize_bleu(bleu_t5_medf, bleu_f200_medf, bleu_hel_medf)
# visualize_ter(ter_t5_medf, ter_f200_medf, ter_hel_medf)
# visualize_meteor(meteor_t5_medf, meteor_f200_medf, meteor_hel_medf)
# plt.show()


# test_df(['Facebook NLLB', 'Helsinki Opus'])


