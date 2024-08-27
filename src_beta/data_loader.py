import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,IterableDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
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
    if args.mode == 'HADE':
        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        
        vad_personality = [eval(i) for i in df['VAD_personality']]
        
        
        if personality == 'A':
            vad_personality = [[i[0]*10, 0.0] for i in vad_personality]
        elif personality == 'C':
            vad_personality = [[i[1]*10, 0.0] for i in vad_personality]
        elif personality == 'E':
            vad_personality = [[i[2]*10, 0.0] for i in vad_personality]
        elif personality == 'O':
            vad_personality = [[i[3]*10, 0.0] for i in vad_personality]
        elif personality == 'N':
            vad_personality = [[i[4]*100, 0.0] for i in vad_personality]
            
        mean_vad_personality = mean([i[0] for i in vad_personality])
        vad_personality = [[i[0]-mean_vad_personality, 0] for i in vad_personality]
        
        
        

        dialog_state = [eval(i) for i in df['dialog_state']]
        
        dialog_state = padding_uttrs(dialog_state, -1, args)
        
        
        emotion = [eval(i) for i in df['Emoberta_softmax']]
        emotion    = padding_uttrs(emotion, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], args)

        labels = list(df['labels'])
        
        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_vad_personality, test_vad_personality,_,_ = train_test_split(vad_personality,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_emotion, test_emotion,_,_ = train_test_split(emotion,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)

        train_dialog_state, test_dialog_state,_,_ = train_test_split(dialog_state,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        

        train_set_labels = train_labels
        
        train_uttrs, valid_uttrs, train_labels, valid_labels = \
            train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        
        train_vad_personality, valid_vad_personality,_,_ = train_test_split(train_vad_personality, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_emotion, valid_emotion,_,_ = train_test_split(train_emotion, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)

        train_dialog_state, valid_dialog_state,_,_ = train_test_split(train_dialog_state, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_uttrs         = torch.tensor(train_uttrs)
        valid_uttrs         = torch.tensor(valid_uttrs)
        test_uttrs          = torch.tensor(test_uttrs)

        train_uttr_masks    = torch.tensor(train_uttr_masks)
        valid_uttr_masks    = torch.tensor(valid_uttr_masks)
        test_uttr_masks     = torch.tensor(test_uttr_masks)


        train_vad_personality        = torch.tensor(train_vad_personality)    
        valid_vad_personality        = torch.tensor(valid_vad_personality)
        test_vad_personality         = torch.tensor(test_vad_personality)

        
        train_emotion        = torch.tensor(train_emotion)    
        valid_emotion        = torch.tensor(valid_emotion)
        test_emotion         = torch.tensor(test_emotion)

        train_dialog_state        = torch.tensor(train_dialog_state)    
        valid_dialog_state        = torch.tensor(valid_dialog_state)
        test_dialog_state         = torch.tensor(test_dialog_state)

        
        train_labels        = torch.tensor(train_labels)    
        valid_labels        = torch.tensor(valid_labels)
        test_labels         = torch.tensor(test_labels)


        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_vad_personality, train_emotion, train_dialog_state, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_vad_personality, valid_emotion, valid_dialog_state, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_vad_personality, test_emotion, test_dialog_state, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length





    elif args.mode == 'Uttr':
        
        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
                pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        labels = list(df['labels'])
        
        train_uttrs, test_uttrs, train_labels, test_labels = \
            train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        
        train_set_labels = train_labels
        
        train_uttrs, valid_uttrs, train_labels, valid_labels = \
            train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        
        train_uttrs         = torch.tensor(train_uttrs)
        valid_uttrs         = torch.tensor(valid_uttrs)
        test_uttrs          = torch.tensor(test_uttrs)

        train_uttr_masks    = torch.tensor(train_uttr_masks)
        valid_uttr_masks    = torch.tensor(valid_uttr_masks)
        test_uttr_masks     = torch.tensor(test_uttr_masks)
        
        train_labels        = torch.tensor(train_labels)    
        valid_labels        = torch.tensor(valid_labels)
        test_labels         = torch.tensor(test_labels)


        train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length
        
    elif args.mode == 'Context':
        '''
        We input the whole dialog into the encoder for personality prediction. 
        We indicated the utterance from the analyzed speaker and the context 
        by segment embeddings in the pre-trained models: 1 for utterances and 0 for dialog context. 
        '''

        uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN/2), \
            pad_to_max_length=True) for sent in df['utterance']]
        uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
        
        contexts = [tokenizer.encode(sent, add_special_tokens=True, max_length=int(args.MAX_LEN/2), \
                pad_to_max_length=True) for sent in df['context']]
        context_masks = [[float(i>0) for i in seq] for seq in contexts]

        sents = []
        sent_masks = []
        sent_seg_embeddings = []

        for i in range(len(uttrs)):
            sents.append(uttrs[i] + contexts[i][1:]) ## remove the latter [CLS]
            sent_masks.append(uttr_masks[i] + context_masks[i][1:])
            sent_seg_embeddings.append([1]*len(uttrs[i]) + [0]*len(contexts[i][1:]))
        
        labels = list(df['labels'])

        train_sents, test_sents, train_labels, test_labels = \
            train_test_split(sents, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
        train_sent_masks, test_sent_masks,_,_ = train_test_split(sent_masks,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)
        train_seg_embeddings, test_seg_embeddings,_,_ = train_test_split(sent_seg_embeddings,labels,random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=labels)


        train_set_labels = train_labels


        train_sents, valid_sents, train_labels, valid_labels = \
            train_test_split(train_sents, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
        train_sent_masks, valid_sent_masks,_,_ = train_test_split(train_sent_masks, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)
        train_seg_embeddings, valid_seg_embeddings,_,_ = train_test_split(train_seg_embeddings, train_set_labels, random_state=args.SEED, \
                                                       test_size=args.test_size,  stratify=train_set_labels)


        train_sents          = torch.tensor(train_sents)
        valid_sents          = torch.tensor(valid_sents)
        test_sents           = torch.tensor(test_sents)

        train_sent_masks     = torch.tensor(train_sent_masks)
        valid_sent_masks     = torch.tensor(valid_sent_masks)
        test_sent_masks      = torch.tensor(test_sent_masks)

        train_seg_embeddings = torch.tensor(train_seg_embeddings)
        valid_seg_embeddings = torch.tensor(valid_seg_embeddings)
        test_seg_embeddings  = torch.tensor(test_seg_embeddings)

        train_labels         = torch.tensor(train_labels)    
        valid_labels         = torch.tensor(valid_labels)
        test_labels          = torch.tensor(test_labels)

        train_data       = TensorDataset(train_sents, train_sent_masks, train_seg_embeddings, train_labels)
        train_sampler    = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        valid_data       = TensorDataset(valid_sents, valid_sent_masks, valid_seg_embeddings, valid_labels)
        valid_sampler    = RandomSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        test_data        = TensorDataset(test_sents, test_sent_masks, test_seg_embeddings, test_labels)
        test_sampler     = RandomSampler(test_data)
        test_dataloader  = DataLoader(test_data, sampler=valid_sampler, batch_size=args.batch_size)
        
        train_length     = len(train_data)
        return train_dataloader, valid_dataloader, test_dataloader, train_length

    elif args.mode == 'Full_dialog':

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
