import zipfile
import os
import pandas as pd
import numpy as np
import re
import torch
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

    train = pd.read_csv('fra.txt', names=['src', 'trg', 'lic'], sep='\t')
    del train['lic']

    train['trg'] = train['trg'].apply(lambda x: clean_text(x))
    train['src'] = train['src'].apply(lambda x: clean_text(x))

    train.to_csv('preprocess.csv', index=False)

if __name__ == '__main__':
    main()