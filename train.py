import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from models import Transformer
from dataset import dataloader
from transformers import AutoTokenizer
import time
import math
from copy import deepcopy
# from tqdm import tqdm
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re

def train(model, iterator, optimizer, criterion, device, clip):
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(enumerate(iterator), total = len(iterator), desc = 'train batch iteration'):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])
        # output: [64 (batch_size), 119 (max_seq_length - 1), 32101 (output_dim)]
        # trg: [64 (batch_size), 120 (max_seq_length)]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        # output: [7616 (batch_size * (max_seq_length - 1)), 32101 (output_dim)]
        # trg: [7616 (batch_size * (trg_len - 1))]

        loss = criterion(output, trg)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        # print('\n' + f'{loss.item()}' + '\n')
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total = len(iterator), desc = 'valid batch iteration'):
            src = batch[0].to(device)
            trg = batch[1].to(device)

            output, _ = model(src, trg[: ,:-1])

            # output: [64(batch_size), 119 (max_seq_length - 1), 32101 (output_dim)]
            # trg: [64 (batch_size), 120 (max_seq_length)]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
    
            # output: [7616 (batch_size * (max_seq_length - 1)), 32101 (output_dim)]
            # trg: [7616 (batch_size * (trg_len - 1))]

            loss = criterion(output, trg)
        
            epoch_loss += loss.item()

    return  epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, type = str)
    config = p.parse_args()
    return config

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    config = define_argparser()
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    tokenizer.add_special_tokens({'bos_token':'<s>'})
    vocab_size = tokenizer.vocab_size + 1
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 120
    dropout = 0.1
    n_epochs = 10
    clip = 1
    early_stopping_patience = 3
    best_valid_accuracy = 0.0
    batch_size =32
    train_split = 0.9
    valid_split = 0.1
    # early_stopping_counter = 0
    

    print('데이터 로드 중...')
    train_dl, valid_dl = dataloader(tokenizer, max_len = max_seq_length, batch_size = batch_size, train_split = train_split, valid_split = valid_split)
    print('데이터 로드 완료')

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = Transformer(tokenizer, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, batch_size, device).to(device)
    if config.model is not None:
        base_directory = './'
        model.load_state_dict(torch.load(base_directory + config.model, weights_only=True))
    else:
        model.apply(initialize_weights)
    # base_directory = './'
    # model.load_state_dict(torch.load(base_directory + 'transformers_english_to_french_20.pt', weights_only=True))


    # learning_rate = 0.0005
    learning_rate = 0.00001
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    best_valid_loss = float('inf')
    # ReduceLROnPlateau 스케줄러 초기화
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=early_stopping_patience, factor=0.1)
    # train the model through multiple epochs
    for epoch in tqdm(range(n_epochs), total = n_epochs, desc = 'Epoch'):
        start_time = time.time()
        print(f'\nEpoch {epoch + 1} 시작: ', time.strftime("%Y년 %m월 %d일 %H시 %M분 %S초", time.localtime(start_time)))
        train_loss = train(model, train_dl, optimizer, criterion, device, clip)
        valid_loss = evaluate(model, valid_dl, criterion, device)

        end_time = time.time()
        print(f'Epoch {epoch + 1} 끝: ', time.strftime("%Y년 %m월 %d일 %H시 %M분 %S초", time.localtime(end_time)))
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # ReduceLROnPlateau 스케줄러에 현재 검증 손실값 전달
        scheduler.step(valid_loss)

        if config.model is not None:
            model_epoch = int(re.sub(r'[^0-9]', '', config.model))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stopping_counter = 0
            
            if config.model is not None:
                print('Saving the best model epoch: ', model_epoch + epoch + 1)
                torch.save(model.state_dict(), f'transformers_english_to_french_{model_epoch + epoch + 1}.pt')
                best_model_epoch = model_epoch + epoch + 1
            else:
                print('Saving the best model epoch: ', epoch + 1)
                torch.save(model.state_dict(), f'transformers_english_to_french_{epoch + 1}.pt')
                best_model_epoch = epoch + 1
            # Store the state_dict of the current best model
            best_model = deepcopy(model.state_dict())
            print("Updated the best model.")
        else:
            early_stopping_counter += 1

        # Check for early stopping based on accuracy
        if scheduler.num_bad_epochs >= scheduler.patience:
        # if early_stopping_counter >= early_stopping_patience:
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
            print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
            print("Early stopping triggered. No improvement for {} consecutive epochs.".format(scheduler.patience))
            if best_model is not None:
                # save the best model
                print('saved the best model')
                torch.save(best_model, f'transformers_english_to_french_{best_model_epoch}_best_model.pt')

            break  # Stop training

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')