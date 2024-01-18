import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from openprompt.data_utils import InputExample

import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainerCallback
import evaluate
from datasets import load_metric
import numpy as np
import time

from openprompt.prompts import ManualTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from openprompt.plms import T5TokenizerWrapper
import torch


def load_training_data(tsv_files, input_sent):
    train_dataset = ''
    valid_dataset = ''
    for tsv_file in tsv_files:
        df = pd.read_csv(tsv_file, sep='\t')
        data_path = '../data/tmp.jsonl'
        class_names = ['no', 'yes']
        
        json_data = df[[input_sent, 'personality_description', 'nli_label']].to_dict(orient="records")
        with open(data_path, 'w') as outfile:
            for row in json_data:
                json.dump(row, outfile)
                outfile.write('\n')
        features = Features({input_sent: Value('string'), 'personality_description': Value('string'),
                     'nli_label': ClassLabel(names=class_names)})
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


def evaluation(eval_dataloader, prompt_model):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(eval_dataloader):
        inputs = inputs.cuda()        
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc


def test(test_tsv, input_sent, tokenizer, mytemplate, WrapperClass, max_seq_length, prompt_model):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    
    df = pd.read_csv(test_tsv, sep='\t')
    test_df = df[[input_sent, 'pos_personality_description', 'neg_personality_description', 'label']]
    cnt = 0
    for i,r in test_df.iterrows():

        dataset = [
            InputExample(text_a=r[input_sent], text_b=r['pos_personality_description'],
                                         label=int(r['label']), guid=0),
            InputExample(text_a=r[input_sent], text_b=r['neg_personality_description'],
                                         label=int(r['label']), guid=1)
        ]
        
        data_loader = PromptDataLoader(dataset=dataset,
            template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
            shuffle=False, teacher_forcing=False,
            decoder_max_length=3,
            predict_eos_token=False,
            truncate_method="head"
        )
        logits_list = []
        for step, inputs in enumerate(data_loader):
            inputs = inputs.cuda()        
            logits = prompt_model(inputs)
            logits_list.append(logits)
        
        alllabels.extend([int(r['label'])])
        allpreds.extend(torch.argmax((logits_list[0]+logits_list[1].flip(dims=[1])), dim=-1).cpu().tolist())
#         allpreds.extend(torch.argmax((logits_list[0]), dim=-1).cpu().tolist())
        
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc




if __name__ == '__main__':
    DATA = 'Friends'
    for SEED in [0,42,3407,1,2,3,4,5,6,7]:
        input_sent = 'affective_dialog' # origin_sent
        template_text = '{"placeholder":"text_a"} ; {"placeholder":"text_b"}? Is it correct? {"mask"}.'

        # model_name = "roberta-base"
        model_name = "t5-base"
        max_seq_length = 256
        use_cuda = True
        learning_rate = 1e-05
        num_epochs = 3
        batch_size = 32

        result_name = DATA + '_pos+neg_prompt_tuning_' + model_name + '_' + input_sent + '_SEED_' + str(SEED) + '.txt'
        result_file = open(result_name, 'w')


        plm, tokenizer, model_config, WrapperClass = load_plm("t5", model_name)
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

        wrapped_tokenizer = WrapperClass(max_seq_length=max_seq_length, tokenizer=tokenizer,truncate_method="head")

        dialog_lens = [1] # 0.25, 0.5, 0.75,
        personality_list = ['A', 'C', 'E', 'O', 'N']
        for dialog_len in dialog_lens:
            result_file.write('Dialog length is ' + str(dialog_len) + '\n')

            for personality in personality_list:
                result_file.write('Personality ' + personality + '\n')

                tsv_files = [
                    '../new_data/'+DATA+'/'+DATA+'_' + personality + '_NLI_' + str(SEED) + '_train.tsv',
                    '../new_data/'+DATA+'/'+DATA+'_' + personality + '_NLI_' + str(SEED) + '_valid.tsv',
                ]

                dataset_dict = load_training_data(tsv_files, input_sent) # train and valid
                dataset = {}
                for split in ['train', 'validation']:
                    dataset[split] = []
                    cnt = 0
                    for data in dataset_dict[split]:
                        input_example = InputExample(text_a=data[input_sent], text_b=data['personality_description'],
                                                     label=int(data['nli_label']), guid=cnt)

                        dataset[split].append(input_example)
                        cnt += 1


                train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
                                                    batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                                    decoder_max_length=3,
                                                    predict_eos_token=False,
                                                    truncate_method="head")

                validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate,
                                                         tokenizer=tokenizer,
                                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
                                                         batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                                         decoder_max_length=3,
                                                         predict_eos_token=False,
                                                         truncate_method="head")

                myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                                                label_words=[["no"], ["yes"]])


                prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                                       freeze_plm=False)
                prompt_model = prompt_model.cuda()

                loss_func = torch.nn.CrossEntropyLoss()
                no_decay = ['bias', 'LayerNorm.weight']
                # it's always good practice to set no decay to biase and LayerNorm parameters
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': 0.01},
                    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0}
                ]

                optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

                best_acc = 0.0
                best_prompt_model = prompt_model
                for epoch in range(num_epochs):
                    tot_loss = 0
                    for step, inputs in enumerate(train_dataloader):
                        inputs = inputs.cuda()
                        logits = prompt_model(inputs)
                        labels = inputs['label'] # From InputExample 

                        loss = loss_func(logits, labels)
                        loss.backward()
                        tot_loss += loss.item()

                        optimizer.step()
                        optimizer.zero_grad()

                        if step % 100 == 1:
    #                         result_file.write("Epoch {}, average training loss: {}\n".format(epoch, tot_loss / (step + 1)))
                            current_acc = evaluation(validation_dataloader, prompt_model)
    #                         result_file.write('Current Evaluation accuracy: ' + str(current_acc) + '\n')
                            if current_acc > best_acc:
                                best_acc = current_acc
                                best_prompt_model = prompt_model
                            print('Current Best Evaluation acc is: ' + str(best_acc) + '\n')
    #                         result_file.write('Current Best Evaluation acc is: ' + str(best_acc) + '\n')

                test_tsv = '../new_data/'+DATA+'/'+DATA+'_' + personality + '_NLI_' + str(SEED) + '_test.tsv'
                test_acc = test(test_tsv, input_sent, tokenizer, mytemplate, WrapperClass, max_seq_length, best_prompt_model)
                result_file.write('Test acc is: ' + str(test_acc) + '\n')
                result_file.write('******************\n')



