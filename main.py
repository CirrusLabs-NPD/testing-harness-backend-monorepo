from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import nltk
import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import pandas as pd
import nltk
from datasets import load_dataset

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
df = pd.read_csv('data/en-fr.csv' , nrows=2000)
df.to_csv("data/base_df.csv", index=False)
clean_df, tokenized_df = clean(df)
print(f"TOKENIZED DF: {tokenized_df}")
clean_df.to_csv("data/clean_df.csv", index=False)

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
clean_medf.to_csv("data/clean_medf.csv", index=False)

def fix_df(df):
    df = df.iloc[1:]
    df.reset_index(drop=True, inplace=True)
    return df

# Get the predictions
predicted_t5_df = pd.read_csv('data/predicted_t5_df.csv', header=None, names=['fr'])
predicted_t5_df = fix_df(predicted_t5_df)

predicted_f200_df = pd.read_csv('data/predicted_f200_df.csv', header=None, names=['fr'])
predicted_f200_df = fix_df(predicted_f200_df)

predicted_t5_medf = pd.read_csv('data/predicted_t5_medf.csv', header=None, names=['fr'])
predicted_t5_medf = fix_df(predicted_t5_medf)

predicted_f200_medf = pd.read_csv('data/predicted_f200_medf.csv', header=None, names=['fr'])
predicted_f200_medf = fix_df(predicted_f200_medf)

# Calculate METEOR scores
def calculate_meteor(tokenized_df, tokenized_predicted):
    meteor_scores = []
    for reference, hypothesis in zip(tokenized_df['fr_tokenized'], tokenized_predicted):
        score = meteor_score.meteor_score([reference], hypothesis)
        meteor_scores.append(score)
    return meteor_scores

# Calculate BLEU score
def calculate_bleu(predictions, references):
    try:
        bleu = evaluate.load("bleu")
        print("Predictions:")
        pprint.pprint(predictions[:5])
        print("\nReferences:")
        pprint.pprint(references[:5])
        bleu_score = bleu.compute(predictions=predictions, references=references)
        return bleu_score
    except Exception as identifier:
        print(f"Got the following error: {identifier}")
        return
    

# Calculate TER score
def calculate_ter(predictions, references):
    try:
        ter = evaluate.load("ter")
        ter_score = ter.compute(predictions=predictions, references=references, ignore_punct=True, case_sensitive=False)
        return ter_score
    except Exception as identifier:
            print(f"Got the following error: {identifier}")
            return

# Test the given translations against all metrics
def test_metrics(clean_df, predicted_df):
    # print(f"PREDICTED_DF: {predicted_df}")
    predicted_tokenized = predicted_df['fr'].apply(nltk.word_tokenize)
    # print(f"PREDICTED_TOKENIZED DF: {predicted_tokenized}")
    predictions = predicted_df['fr'].tolist()
    references = []
    for value in clean_df['fr']:
        references.append(value)
    
    clean_df['predicted'] = predicted_df['fr']
    clean_df['meteor_score'] = calculate_meteor(tokenized_df, predicted_tokenized)
    print(f"CLEAN_DF: {clean_df}")
    bleu_score = calculate_bleu(predictions, references)
    ter_score = calculate_ter(predictions, references)
    
    return clean_df, bleu_score, ter_score

# Test the datasets against the t5 model
def test_t5():
    final_t5_df, bleu_t5_df, ter_t5_df = test_metrics(clean_df, predicted_t5_df)
    final_t5_medf, bleu_t5_medf, ter_t5_medf = test_metrics(clean_medf, predicted_t5_medf)
    return

# Test the datasets against the nllb-200 model
def test_f200():
    final_f200_df, bleu_f200_df, ter_f200_df = test_metrics(clean_df, predicted_f200_df)
    final_f200_medf, bleu_f200_medf, ter_f200_medf = test_metrics(clean_medf, predicted_f200_medf)
    return

# Test the models on the default dataset
def test_df():
    final_t5_df, bleu_t5_df, ter_t5_df = test_metrics(clean_df, predicted_t5_df)
    print(f"FINAL T5 DF: {final_t5_df}")
    print(f"T5 BLEU: {bleu_t5_df}")
    print(f"T5 TER: {ter_t5_df}")
    
    # F200 METRICS AFFECT T5 METEOR SCORES FIX IF YOU CAN PLZ
    
    final_f200_df, bleu_f200_df, ter_f200_df = test_metrics(clean_df, predicted_f200_df)
    print(f"FINAL f200 DF: {final_f200_df}")
    print(f"f200 BLEU: {bleu_f200_df}")
    print(f"f200 TER: {ter_f200_df}")
    return

# Test the models on the medical dataset
def test_medf():
    final_t5_medf, bleu_t5_medf, ter_t5_medf = test_metrics(clean_medf, predicted_t5_medf)
    final_f200_medf, bleu_f200_medf, ter_f200_medf = test_metrics(clean_medf, predicted_f200_medf)
    return

test_df()

    

# clean_df['en_len'] = clean_df['en'].apply(len)
# plt.scatter(clean_df['en_len'], clean_df['meteor_score'])
# plt.show()