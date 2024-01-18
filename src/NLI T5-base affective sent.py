import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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


def load_data(tsv_file, input_sent):
    df = pd.read_csv(tsv_file, sep='\t')
    if input_sent == 'affective_sent':
        df[input_sent] = df['origin_sent'] + ' <s> ' + df['affective_prompt']
    data_path = '../data/tmp.jsonl'
    json_data = df[[input_sent, 'personality_description', 'labels']].to_dict(orient="records")
    with open(data_path, 'w') as outfile:
        for row in json_data:
            json.dump(row, outfile)
            outfile.write('\n')

    class_names = ['no', 'yes']
    features = Features({input_sent: Value('string'), 'personality_description': Value('string'),
                         'labels': ClassLabel(names=class_names)})
    dataset_dict = load_dataset("json", data_files=data_path, features=features)

    tmp_dict = dataset_dict['train'].train_test_split(test_size=0.2, shuffle=True, seed=SEED)
    train_dataset, remaining_dataset = tmp_dict['train'], tmp_dict['test']
    tmp_dict = remaining_dataset.train_test_split(test_size=0.5, shuffle=True, seed=SEED)
    valid_dataset, test_dataset = tmp_dict['train'], tmp_dict['test']
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
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


if __name__ == '__main__':

    SEED = 42
    input_sent = 'origin_sent' # origin_sent affective_sent
    template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    model_name = "t5-base"
    max_seq_length = 128
    use_cuda = True
    learning_rate = 1e-5
    num_epochs = 3
    result_name = 'result_' + model_name + '_' + input_sent + '_' + str(learning_rate) + '.txt'

    result_file = open(result_name, 'w')

    dialog_lens = [1] # 0.25, 0.5, 0.75,
    personality_list = ['A', 'C', 'E', 'O', 'N']
    for dialog_len in dialog_lens:
        result_file.write('Dialog length is ' + str(dialog_len) + '\n')

        for personality in personality_list:
            result_file.write('Personality ' + personality + '\n')

            tsv_file = '../new_data/CPED_' + personality + '_with_role_' + str(dialog_len) + '.tsv'
            dataset_dict = load_data(tsv_file, input_sent)
            dataset = {}
            for split in ['train', 'validation', 'test']:
                dataset[split] = []
                cnt = 0
                for data in dataset_dict[split]:
                    input_example = InputExample(text_a=data[input_sent], text_b=data['personality_description'],
                                                 label=int(data['labels']), guid=cnt)
                    dataset[split].append(input_example)
                    cnt += 1

            plm, tokenizer, model_config, WrapperClass = load_plm("t5", model_name)

            mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
            wrapped_t5tokenizer = T5TokenizerWrapper(max_seq_length=max_seq_length, decoder_max_length=3,
                                                     tokenizer=tokenizer, truncate_method="head")

            model_inputs = {}
            for split in ['train', 'validation', 'test']:
                model_inputs[split] = []
                for sample in dataset[split]:
                    tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample),
                                                                                 teacher_forcing=False)
                    model_inputs[split].append(tokenized_example)

            train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                                tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                                decoder_max_length=3,
                                                batch_size=4, shuffle=True, teacher_forcing=False,
                                                predict_eos_token=False,
                                                truncate_method="head")

            validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate,
                                                     tokenizer=tokenizer,
                                                     tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                                     decoder_max_length=3,
                                                     batch_size=4, shuffle=False, teacher_forcing=False,
                                                     predict_eos_token=False,
                                                     truncate_method="head")

            test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                               tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                               decoder_max_length=3,
                                               batch_size=4, shuffle=False, teacher_forcing=False,
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
                    labels = inputs['label']

                    loss = loss_func(logits, labels)
                    loss.backward()
                    tot_loss += loss.item()

                    optimizer.step()
                    optimizer.zero_grad()

                    if step % 100 == 1:
                        result_file.write("Epoch {}, average training loss: {}\n".format(epoch, tot_loss / (step + 1)))
                        current_acc = evaluation(validation_dataloader, prompt_model)
                        result_file.write('Current Evaluation accuracy: ' + str(current_acc) + '\n')
                        if current_acc > best_acc:
                            best_acc = current_acc
                            best_prompt_model = prompt_model
                        print('Current Best Evaluation acc is: ' + str(best_acc) + '\n')
                        result_file.write('Current Best Evaluation acc is: ' + str(best_acc) + '\n')

            test_acc = evaluation(test_dataloader, best_prompt_model)
            result_file.write('Test acc is: ' + str(test_acc) + '\n')
            result_file.write('******************\n')

    result_file.close()




