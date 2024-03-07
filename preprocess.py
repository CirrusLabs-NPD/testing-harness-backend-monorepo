import pandas as pd
import nltk
from datasets import load_dataset



def clean(df):
    # Lowercase
    df['en'] = df['en'].str.lower()
    df['fr'] = df['fr'].str.lower()

    # Filter by length
    df = df[(df['en'].str.len() <= 50) & (df['fr'].str.len() <= 50)]
    df.reset_index(drop=True, inplace=True)

    df_tokenized = pd.DataFrame()

    # Tokenize sentences
    df_tokenized['en_tokenized'] = df['en'].apply(nltk.word_tokenize)
    df_tokenized['fr_tokenized'] = df['fr'].apply(nltk.word_tokenize)

    print(df.head())
    return df, df_tokenized

df = pd.read_csv('data/en-fr.csv' , nrows=2000)
df, df_tokenized = clean(df)

df.to_csv("data/clean_df.csv", index=False)
df_tokenized.to_csv("data/tokenized_df.csv", index=False)

dataset = load_dataset('qanastek/WMT-16-PubMed', split=['train[0:5000]'])
temp = pd.DataFrame(dataset)
temp = temp.transpose()

en_column = []
fr_column = []
print(temp)
for sentence in temp[0]:
        en = sentence['translation']['en']
        fr = sentence['translation']['fr']
        
        en_column.append(en)
        fr_column.append(fr)

medf = pd.DataFrame()
medf['en'] = en_column
medf['fr'] = fr_column
print(medf)

clean_medf, tokenized_medf = clean(medf)
clean_medf.to_csv("data/clean_medf.csv", index=False)
tokenized_medf.to_csv("data/tokenized_medf.csv", index=False)