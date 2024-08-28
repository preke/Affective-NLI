import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainerCallback
import evaluate
from datasets import load_metric
import numpy as np
import time

from transformers import AutoModelForSequenceClassification
from peft import PeftModelForSequenceClassification, get_peft_config
from transformers import AdamW, get_linear_schedule_with_warmup

import torch


def load_training_data(tsv_files, input_sent):
    train_dataset = ''
    valid_dataset = ''
    for tsv_file in tsv_files:
        df = pd.read_csv(tsv_file, sep='\t')
        df['input'] = df[input_sent] + ' ; ' + df['personality_description'] + '; Is it correct? '
        data_path = '../data/tmp.jsonl'
        class_names = ['no', 'yes']
        
        json_data = df[['input', 'nli_label']].to_dict(orient="records")
        with open(data_path, 'w') as outfile:
            for row in json_data:
                json.dump(row, outfile)
                outfile.write('\n')
        features = Features({'input': Value('string'), 'nli_label': ClassLabel(names=class_names)})
        dataset_dict = load_dataset("json", data_files=data_path, features=features)
        if 'train' in tsv_file:
            train_dataset = dataset_dict['train']
        elif 'valid' in tsv_file:
            valid_dataset = dataset_dict['train']
        
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
    })
    
    return dataset_dict



def evaluation(eval_dataloader, peft_model):
    peft_model.eval()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(eval_dataloader):
        input_ids = torch.stack(inputs['input_ids']).T.cuda()
        attn_masks = torch.stack(inputs['attention_mask']).T.cuda()
        logits = peft_model(input_ids, attn_masks).logits.cpu()
        labels = inputs['nli_label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc


def test(test_tsv, input_sent, tokenizer, max_seq_length, prompt_model):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    
    df = pd.read_csv(test_tsv, sep='\t')
    test_df = df[[input_sent, 'pos_personality_description', 'neg_personality_description', 'label']]

    cnt = 0
    
    for i,r in test_df.iterrows():
        pos_sample = r[input_sent] + ' ; ' + r['pos_personality_description'] + '; Is it correct? '
        neg_sample = r[input_sent] + ' ; ' + r['neg_personality_description'] + '; Is it correct? '
        label = r['label']
        
        tokenized_pos = tokenizer(pos_sample, truncation=True, max_length=256, pad_to_max_length=True)
        tokenized_neg = tokenizer(neg_sample, truncation=True, max_length=256, pad_to_max_length=True)
        
        logits_list = []
        
        input_ids = torch.LongTensor(tokenized_pos['input_ids']).unsqueeze(0).cuda()
        attn_masks = torch.LongTensor(tokenized_pos['attention_mask']).unsqueeze(0).cuda()
        # print(input_ids, attn_masks)
        pos_logits = peft_model(input_ids, attn_masks).logits.cpu()

        input_ids = torch.LongTensor(tokenized_neg['input_ids']).unsqueeze(0).cuda()
        attn_masks = torch.LongTensor(tokenized_neg['attention_mask']).unsqueeze(0).cuda()
        neg_logits = peft_model(input_ids, attn_masks).logits.cpu()
        
        alllabels.extend([int(r['label'])])
        allpreds.extend(torch.argmax((pos_logits+neg_logits.flip(dims=[1])), dim=-1).cpu().tolist())
#         allpreds.extend(torch.argmax((logits_list[0]), dim=-1).cpu().tolist())
        
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["input"], truncation=True, max_length=256, pad_to_max_length=True)
    return outputs

def compute_metrics(eval_pred):
    metric     = evaluate.load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    metric     = evaluate.load('accuracy')
    DATA = 'Friends'
    
    SEED_list = [0,42,3407,1,2,3,4,5,6,7]
    dialog_lens = [1, 0.25, 0.5, 0.75]
    personality_list = ['A', 'C', 'E', 'O', 'N']
    
    for SEED in SEED_list:
        input_sent = 'affective_dialog' 
        

        model_name = "meta-llama/Llama-2-7b-chat-hf"
        max_seq_length = 256
        use_cuda = True
        learning_rate = 1e-05
        num_epochs = 10
        batch_size = 3

        result_name = DATA + '_pos+neg_prompt_tuning_llama_' + input_sent + '_SEED_' + str(SEED) + '.txt'
        result_file = open(result_name, 'w')

        
        for dialog_len in dialog_lens:
            result_file.write('Dialog length is ' + str(dialog_len) + '\n')

            for personality in personality_list:
                result_file.write('Personality ' + personality + '\n')


                config = {
                    "peft_type": "PREFIX_TUNING",
                    "task_type": "SEQ_CLS",
                    "inference_mode": False,
                    "num_virtual_tokens": 20,
                    "prefix_projection": False,
                }
                
                peft_config = get_peft_config(config)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype="auto")
                model.config.pad_token_id = model.config.eos_token_id
                peft_model = PeftModelForSequenceClassification(model, peft_config)
                peft_model.print_trainable_parameters()
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
                tsv_files = [
                   '../new_data/'+DATA+'_' + personality + '_with_role_' +str(dialog_len)+ '_' + str(SEED) + '_train.tsv',
                   '../new_data/'+DATA+'_' + personality + '_with_role_' +str(dialog_len)+ '_' + str(SEED) + '_valid.tsv',
                ]

                dataset = load_training_data(tsv_files, input_sent) # train and valid
                

                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["input"],
                )
                
                train_dataloader = DataLoader(dataset=tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
                
                validation_dataloader = DataLoader(dataset=tokenized_datasets["validation"], batch_size=batch_size, shuffle=True)


                peft_model = peft_model.cuda()

                loss_func = torch.nn.CrossEntropyLoss()
                no_decay = ['bias', 'LayerNorm.weight']
                # it's always good practice to set no decay to biase and LayerNorm parameters
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in peft_model.named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': 0.01},
                    {'params': [p for n, p in peft_model.named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0}
                ]

                optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

                best_acc = 0.0
                best_prompt_model = peft_model
                for epoch in range(num_epochs):
                    tot_loss = 0
                    for step, inputs in enumerate(train_dataloader):
                        input_ids = torch.stack(inputs['input_ids']).T.cuda()
                        attn_masks = torch.stack(inputs['attention_mask']).T.cuda()
                        logits = peft_model(input_ids, attn_masks).logits.cpu()
                        labels = inputs['nli_label']

                        loss = loss_func(logits, labels)
                        loss.backward()
                        tot_loss += loss.item()

                        optimizer.step()
                        optimizer.zero_grad()

                        if step % 100 == 1:
#                             result_file.write("Epoch {}, average training loss: {}\n".format(epoch, tot_loss / (step + 1)))
                            current_acc = evaluation(validation_dataloader, peft_model)
#                             result_file.write('Current Evaluation accuracy: ' + str(current_acc) + '\n')
                            if current_acc > best_acc:
                                best_acc = current_acc
                                best_prompt_model = peft_model
#                             print('Current Best Evaluation acc is: ' + str(best_acc) + '\n')
    #                         result_file.write('Current Best Evaluation acc is: ' + str(best_acc) + '\n')
                
                

                test_tsv =  '../new_data/'+DATA+'_' + personality + '_with_role_' +str(dialog_len)+ '_' + str(SEED) + '_test.tsv'
        
                test_acc = test(test_tsv, input_sent, tokenizer, max_seq_length, peft_model)
                result_file.write('Test acc is: ' + str(test_acc) + '\n')
                result_file.write('******************\n')
                # '''


