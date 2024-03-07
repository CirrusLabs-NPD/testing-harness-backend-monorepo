from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import nltk
import evaluate
import matplotlib.pyplot as plt
import pandas as pd

# nltk.download('wordnet')

df = pd.read_csv('data/clean_data.csv')
df_tokenized = pd.read_csv('data/tokenized_data.csv')
df_predicted = pd.read_csv('data/predicted_data.csv')

# f200_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# f200_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# ti_tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.1")
# ti_model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-7B-v0.1")

# Calculate METEOR scores
def calculate_meteor(df_tokenized, tokenized_predicted):
    meteor_scores = []
    for reference, hypothesis in zip(df_tokenized['fr_tokenized'], tokenized_predicted):
        score = meteor_score.meteor_score([reference], hypothesis)
        meteor_scores.append(score)
    return meteor_scores

# Calculate BLEU score
def calculate_bleu(df, predictions, references):
    # Load the BLEU evaluation metric
    bleu = evaluate.load("bleu")

    
    for value in df['fr']:
        references.append(value)

    bleu_scores = []
    score = bleu.compute(predictions=predictions, references=references)
    bleu_scores.append(score)
    return bleu_scores

# Calculate TER score
def calculate_ter(df, predictions, references):
    ter = evaluate.load("ter")
    ter_score = ter.compute(predictions=predictions, references=references, ignore_punct=True, case_sensitive=False)
    return ter_score




# Tokenize each hypothesis sentence
predicted_tokenized = df_predicted.apply(nltk.word_tokenize)
print(predicted_tokenized)

predictions = df_predicted
references = []

# CALCULATE METEOR SCORES
df['predicted'] = df_predicted
df['meteor_score'] = calculate_meteor(df_tokenized, predicted_tokenized)
# Print the dataset with METEOR scores
print(df)

# CALCULATE BLEU SCORE
bleu_scores = calculate_bleu(df, predictions, references)
print("BLEU SCORES: ")
print(bleu_scores)

# CALCULATE TER SCORE
ter_score = calculate_ter(df, predictions, references)
print("TER SCORES: ")
print(ter_score)

# Print the dataset with scores
print(df)

df['en_len'] = df['en'].apply(len)
plt.scatter(df['en_len'], df['meteor_score'])
plt.show()