import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,IterableDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import json
import re
from statistics import mean 


# from EmoBERTa
emotions = {
    "neutral"  : [0.0, 0.0, 0.0], #0.0,
    "joy"      : [0.76, 0.48, 0.35], #1.0,
    "surprise" : [0.4, 0.67, -0.13], #2.0,
    "anger"    : [-0.43, 0.67, 0.34], #3.0,
    "sadness"  : [-0.63, 0.27, -0.33], #4.0,
    "disgust"  : [-0.6, 0.35, 0.11], #5.0,
    "fear"     : [-0.64, 0.6, -0.43]  #6.0
}


def get_vad(VAD_dict, sents, tokenizer, dialog_emo_label):

    dialog_vad = [emotions[i] for i in eval(dialog_emo_label)]
    # dialog_vad = [[0.0, 0.0, 0.0] for i in eval(dialog_emo_label)]

    cnt = 0
    VAD_scores = []
    for sent in sents:
        w_list = re.sub(r'[^\w\s\[\]]','',tokenizer.decode(sent)).split()
        v_score, a_score, d_score = 0, 0, 0
        for word in w_list:
            try:
                v_score += VAD_dict[word][0]
                a_score += VAD_dict[word][1]
                d_score += VAD_dict[word][2]
            except:
                v_score += 0
                a_score += 0
                d_score += 0
        v_score /= float(len(w_list))
        a_score /= float(len(w_list))
        d_score /= float(len(w_list))
        
#         v_score += dialog_vad[cnt][0]
#         a_score += dialog_vad[cnt][1]
#         d_score += dialog_vad[cnt][2]

        cnt += 1

        VAD_scores.append([v_score, a_score, d_score])
    return VAD_scores

def get_VAD_tokenized_dict(i, VAD_tokenized_dict):
    try:
        return VAD_tokenized_dict[i]
    except:
        return [0.0, 0.0, 0.0]



def padding_uttrs(contexts, padding_element, args):
    ans_contexts = []
    
    for sents in contexts:
        pad_num = args.MAX_NUM_UTTR - len(sents) 
        if pad_num > 0: # e.g. 30 - 15 
            for i in range(pad_num):
                sents.append(padding_element)
        elif pad_num < 0: # e.g. 30 - 36 
            sents = sents[:args.MAX_NUM_UTTR]
        ans_contexts.append(sents)

    return ans_contexts


def load_data(df, args, tokenizer, personality):

    if args.mode == 'Full_dialog':

        dialogs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['sent']]
        dialog_masks = [[float(i>0) for i in seq] for seq in dialogs]
        
        labels = list(df['labels'])
        
        train_dialogs, test_dialogs, train_labels, test_labels = \
            train_test_split(dialogs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_dialog_masks, test_dialog_masks,_,_ = train_test_split(dialog_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_set_labels = train_labels
        
        train_dialogs, valid_dialogs, train_labels, valid_labels = \
            train_test_split(train_dialogs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_dialog_masks, valid_dialog_masks,_,_ = train_test_split(train_dialog_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_dialogs         = torch.tensor(train_dialogs)
        valid_dialogs         = torch.tensor(valid_dialogs)
        test_dialogs          = torch.tensor(test_dialogs)

        train_dialog_masks    = torch.tensor(train_dialog_masks)
        valid_dialog_masks    = torch.tensor(valid_dialog_masks)
        test_dialog_masks     = torch.tensor(test_dialog_masks)
        
        train_labels        = torch.tensor(train_labels)    
        valid_labels        = torch.tensor(valid_labels)
        test_labels         = torch.tensor(test_labels)


        train_data       = TensorDataset(train_dialogs, train_dialog_masks, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_dialogs, valid_dialog_masks, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_dialogs, test_dialog_masks, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

