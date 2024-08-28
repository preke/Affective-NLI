# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback

import sys
sys.path.append('../src')


from data_loader import load_data
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# CONFIG

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()
args.device        = 0
args.MAX_LEN       = 128
args.batch_size    = 32
args.adam_epsilon  = 1e-8
args.epochs        = 3
args.num_class     = 7
args.test_size     = 0.1





personalities = ['A','C','E','O','N']


tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base").cuda(args.device)
df = pd.read_csv('Friends_'+personalities[0]+'_whole.tsv', sep='\t') # only need labeling once


# uttrs for the speaker

uttrs      = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True) for sent in df['utterance']]
uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]

uttrs      = torch.tensor(uttrs)
uttr_masks = torch.tensor(uttr_masks)

data       = TensorDataset(uttrs, uttr_masks)
sampler    = RandomSampler(data)
dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size, shuffle=False)


with open('uttr_emo_labels.txt', 'w') as f:
    uttr_pred_list = np.array([])
    model.eval()
    for batch in dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)
        with torch.no_grad():
            b_uttrs, b_uttr_masks = batch
            outputs   = model(b_uttrs, attention_mask=b_uttr_masks)
            logits    = outputs.logits
            logits    = logits.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            uttr_pred_list = np.append(uttr_pred_list, pred_flat)
    f.write(str(list(uttr_pred_list)))
print(uttr_pred_list)
print('*'*10)

### all sents


dialogues    = df['raw_text'].apply(lambda x: [i[1] for i in eval(x)])

dialogs = []
for sents in dialogues:
    dialogs += sents

dialogs      = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True) for sent in dialogs]
dialog_masks = [[float(i>0) for i in seq] for seq in dialogs]

# dialogs       = [[tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, pad_to_max_length=True) for sent in sents] for sents in dialogues]
# dialog_masks  = [[[float(i>0) for i in seq] for seq in sents] for sents in dialogs]

dialogs      = torch.tensor(dialogs)
dialog_masks = torch.tensor(dialog_masks)


data       = TensorDataset(dialogs, dialog_masks)
sampler    = RandomSampler(data)
dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size, shuffle=False)

with open('dialog_emo_labels.txt', 'w') as f:
    dialogs_pred_list = np.array([])
    model.eval()
    for batch in dataloader:
        batch     = tuple(t.cuda(args.device) for t in batch)
        with torch.no_grad():
            b_dialogs, b_dialog_masks = batch
            outputs   = model(b_dialogs, attention_mask=b_dialog_masks)
            logits    = outputs.logits
            logits    = logits.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            dialogs_pred_list = np.append(dialogs_pred_list, pred_flat)
    f.write(str(list(dialogs_pred_list)))
print(dialogs_pred_list)



## --------------

with open('uttr_emo_labels.txt', 'r') as u_f:
    uttr_label_list = eval(u_f.readline())

with open('dialog_emo_labels.txt', 'r') as d_f:
    dialog_label_list = eval(d_f.readline())
    

personality = personalities[0]
file_name = 'Friends_'+personality+'_whole.tsv'
df_A = pd.read_csv(file_name, sep='\t')

len_of_dialog = df_A['dialog_state'].apply(lambda x: len(eval(x)))


emotions = {'0.0': "neutral",
            '1.0': "joy",
            '2.0': "surprise",
            '3.0': "anger",
            '4.0': "sadness",
            '5.0': "disgust",
            '6.0': "fear"}


cnt = 0
dialog_labels = []
for l in len_of_dialog:
    dialog_labels.append(dialog_label_list[cnt:cnt + l])
    cnt += l
    
for personality in personalities:
    file_name = 'Friends_'+personality+'_whole.tsv'
    df = pd.read_csv(file_name, sep='\t')
    df['Uttr_EmoBERTa_label'] = [emotions[str(i)] for i in uttr_label_list]
    df['Dialog_EmoBERTa_label'] = [[emotions[str(i)] for i in j] for j in dialog_labels]
    df.to_csv(file_name, sep='\t')
    
    
    
    



