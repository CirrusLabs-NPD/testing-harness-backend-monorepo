import nltk
import matplotlib
import pandas as pd
import pandas as pd
import nltk
import asyncio
from datasets import load_dataset
import os
import pathlib
from .translation import translate_dataset
import dask
from .metrics import test_metrics, save_metrics, clean_scores, print_bleu_ter, print_meteor

matplotlib.use('agg')

# Define paths to data and result folders
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
UPLOADED_FOLDER = os.path.join(os.path.dirname(__file__), 'uploaded')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')

# nltk.download('wordnet')

# Preprocess the given df
def clean(df):
    #Assume [0] is English and [1] is French
    
    clean_df = pd.DataFrame()
    
    # Lowercase
    clean_df['en'] = df.iloc[:, 0].str.lower()
    clean_df['fr'] = df.iloc[:, 1].str.lower()

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

# Prepare list of datasets
def prep_dataset(dataset_name):
    if dataset == 'Standard':
        return clean_df, tokenized_df
    elif dataset == 'Medical':
        return clean_medf, tokenized_medf
    else:
        ext = pathlib.Path(dataset_name).suffix
        if ext == ".csv":
            df = pd.read_csv(os.path.join(UPLOADED_FOLDER, f'{dataset_name}.csv') , nrows=5000)
            clean_df, tokenized_df = clean(df)
            print(f"TOKENIZED DF: {tokenized_df}")
            clean_df.to_csv(os.path.join(DATA_FOLDER, f'{dataset_name}.csv') , index=False)
            tokenized_df.to_csv(os.path.join(DATA_FOLDER, f'{dataset_name}_tokenized.csv') , index=False)
            return clean_df, tokenized_df
        else:
            dataset = load_dataset(f'{dataset_name}', split=['train[0:5000]'], trust_remote_code=True)
            df = pd.DataFrame(dataset)
            clean_df, tokenized_df = clean(df)
            name = dataset_name.rsplit('/', 1)[-1]
            clean_df.to_csv(os.path.join(DATA_FOLDER, f'{name}.csv'), index=False)
            tokenized_df.to_csv(os.path.join(DATA_FOLDER, f'{name}_tokenized.csv') , index=False)
            return clean_df, tokenized_df

# Fix medical dataset
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

# Current method of testing given datasets and models
def test_custom(selected_datasets, selected_models):
    bleu_results = {}
    ter_results = {}
    meteor_results = {}
    accuracy_results = {}
    for dataset in selected_datasets:
        for model in selected_models:
            if dataset == 'Standard':
                if model == 'Google T5':
                    bleu_t5_df, ter_t5_df, meteor_t5_df, accuracy_t5_df = test_metrics(clean_df, tokenized_df, predicted_t5_df)
                    meteor_results.setdefault(dataset, []).append(meteor_t5_df)
                    bleu_results.setdefault(dataset, []).append(bleu_t5_df)
                    ter_results.setdefault(dataset, []).append(ter_t5_df)
                    accuracy_results.setdefault(dataset, []).append(accuracy_t5_df)
                if model == 'Facebook NLLB':
                    bleu_f200_df, ter_f200_df, meteor_f200_df, accuracy_f200_df = test_metrics(clean_df, tokenized_df, predicted_f200_df)
                    meteor_results.setdefault(dataset, []).append(meteor_f200_df)
                    bleu_results.setdefault(dataset, []).append(bleu_f200_df)
                    ter_results.setdefault(dataset, []).append(ter_f200_df)
                    accuracy_results.setdefault(dataset, []).append(accuracy_f200_df)
                if model == 'Helsinki Opus':
                    bleu_hel_df, ter_hel_df, meteor_hel_df, accuracy_hel_df = test_metrics(clean_df, tokenized_df, predicted_hel_df)
                    meteor_results.setdefault(dataset, []).append(meteor_hel_df)
                    bleu_results.setdefault(dataset, []).append(bleu_hel_df)
                    ter_results.setdefault(dataset, []).append(ter_hel_df)
                    accuracy_results.setdefault(dataset, []).append(accuracy_hel_df)
            elif dataset == 'Medical':
                if model == 'Google T5':
                    bleu_t5_medf, ter_t5_medf, meteor_t5_medf, accuracy_t5_medf = test_metrics(clean_medf, tokenized_medf, predicted_t5_medf)
                    meteor_results.setdefault(dataset, []).append(meteor_t5_medf)
                    bleu_results.setdefault(dataset, []).append(bleu_t5_medf)
                    ter_results.setdefault(dataset, []).append(ter_t5_medf)
                    accuracy_results.setdefault(dataset, []).append(accuracy_t5_medf)
                if model == 'Facebook NLLB':
                    bleu_f200_medf, ter_f200_medf, meteor_f200_medf, accuracy_f200_medf = test_metrics(clean_medf, tokenized_medf, predicted_f200_medf)
                    meteor_results.setdefault(dataset, []).append(meteor_f200_medf)
                    bleu_results.setdefault(dataset, []).append(bleu_f200_medf)
                    ter_results.setdefault(dataset, []).append(ter_f200_medf)
                    accuracy_results.setdefault(dataset, []).append(accuracy_f200_medf)
                if model == 'Helsinki Opus':
                    bleu_hel_medf, ter_hel_medf, meteor_hel_medf, accuracy_hel_medf = test_metrics(clean_medf, tokenized_medf, predicted_hel_medf)
                    meteor_results.setdefault(dataset, []).append(meteor_hel_medf)
                    bleu_results.setdefault(dataset, []).append(bleu_hel_medf)
                    ter_results.setdefault(dataset, []).append(ter_hel_medf)
                    accuracy_results.setdefault(dataset, []).append(accuracy_hel_medf)
    
    print(bleu_results)
    colors = {'Standard': 'blue', 'Medical': 'green'}
    bleu_image = print_bleu_ter("bleu", colors, bleu_results, selected_datasets, selected_models)
    ter_image = print_bleu_ter("ter", colors, ter_results, selected_datasets, selected_models)
    # meteor_image = print_meteor("meteor", meteor_results, selected_datasets, selected_models)
    accuracy_image = print_bleu_ter("accuracy", colors, accuracy_results, selected_datasets, selected_models)
    
    return bleu_image, ter_image, accuracy_image

# def test_user(selected_datasets, selected_models):
#     if dataset == 'Standard':
#         return test_df(selected_models)
#     elif dataset == 'Medical':
#         return test_medf(selected_models)


# Process a given chunk to get image results
def process_chunk(chunk, tokenized_chunk, selected_models):
    predictions = translate_dataset(chunk, selected_models)
    final_results = []
    meteor_results = []
    bleu_results = []
    ter_results = []
    accuracy_results = []
    
    for predicted_df in predictions:
        final_model_df, bleu_model_df, ter_model_df, meteor_model_df, accuracy_model_df = test_metrics(chunk, tokenized_chunk, predicted_df)
        final_results.append(final_model_df)
        meteor_results.append(meteor_model_df)
        bleu_results.append(bleu_model_df)
        ter_results.append(ter_model_df)
        accuracy_results.append(accuracy_model_df)
    
    bleu_image = print_bleu_ter("bleu", "blue", bleu_results, selected_models)
    ter_image = print_bleu_ter("ter", "orange", ter_results, selected_models)
    meteor_image = print_meteor("meteor", meteor_results, selected_models)
    accuracy_image = print_bleu_ter("accuracy", "green", accuracy_results, selected_models)
    
    return bleu_image, ter_image, meteor_image, accuracy_image

# Async process a list of given datasets with the given models
async def generate_test_predictions(selected_datasets, selected_models):
    clean_datasets = []
    tokenized_datasets = []
    for dataset in selected_datasets:
        clean_df, tokenized_df = prep_dataset(dataset)
        clean_datasets.append(clean_df)
        tokenized_datasets.append(tokenized_df)

    tasks = []
    for i in range(len(clean_datasets)):
        df_rows = len(clean_datasets[i])
        max_chunk_size = 50
        num_chunks = (df_rows + max_chunk_size - 1) // max_chunk_size
        dfs = dask.compute(*clean_datasets[i].random_split(n=num_chunks))

        for df_chunk, tokenized_df_chunk in zip(dfs, tokenized_datasets[i].random_split(n=num_chunks)):
            task = asyncio.create_task(process_chunk(df_chunk, tokenized_df_chunk, selected_models))
            tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results



#Fix:
# async
