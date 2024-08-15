import zipfile
import os
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer
from models import Transformer

# 정규표현식 활용하여 데이터 전처리
def clean_text(inputString):
    text_rmv = re.sub(r'[\\\xa0·«»]', '', inputString)
    # 다수 개의 공백을 하나의 공백으로 치환
    text_rmv = re.sub(r"\s+", " ", text_rmv)
    return text_rmv

def main():
    filename = 'fra-eng.zip'
    path = os.getcwd()
    zipfilename = os.path.join(path, filename)

    with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
        zip_ref.extractall(path)

    df = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
    del df['lic']

    df['tar'] = df['tar'].apply(lambda x: clean_text(x))
    # df['tar'] = df['tar'].apply(lambda x: x.lower())
    df['src'] = df['src'].apply(lambda x: clean_text(x))
    # df['src'] = df['src'].apply(lambda x: x.lower())

    total_len = len(df)
    idx = np.random.permutation(total_len)
    split = int(total_len*0.9)
    train_df = df.iloc[idx[:split]]
    train_df.reset_index(inplace=True, drop=True)
    test_df = df.iloc[idx[split:]]
    test_df.reset_index(inplace=True, drop=True)

    train_df.to_csv('train_preprocess.csv', index=False)
    test_df.to_csv('test_preprocess.csv', index=False)

if __name__ == '__main__':
    main()