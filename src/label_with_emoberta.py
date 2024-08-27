# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
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





personalities = ['A']#,'C','E','O','N']


tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
df = pd.read_csv('../data/Friends_'+personalities[0]+'_whole.tsv', sep='\t')
model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base").cuda(args.device)



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







