import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import evaluate
from datasets import load_metric
import numpy as np

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
checkpoint = "roberta-large"
# checkpoint = "microsoft/deberta-v3-large"
# checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='right')
SEED = 42
metric = evaluate.load('f1')

def load_data(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    data_path = '../data/tmp.jsonl'
    json_data = df[['sent', 'labels']].to_dict(orient="records")
    with open(data_path, 'w') as outfile:
        for row in json_data:
            json.dump(row, outfile)
            outfile.write('\n')

    class_names = ['negative', 'positive']
    features = Features({'sent': Value('string'), 'labels': ClassLabel(names=class_names)})
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


def tokenize_function(example):
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer(example["sent"], truncation=True, max_length=256)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def training(data, mode):


    dataset_dict = load_data(data)
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=['sent'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")



    if mode == 'fine-tuning':
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

        training_args = TrainingArguments(
            'test_trainer',
            overwrite_output_dir=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            learning_rate=5e-05,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            seed=SEED,
        )


    elif mode == 'p-tuning':

        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, return_dict=True)

        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=20,
            encoder_hidden_size=128
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir="peft-p-tuning",
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            seed=SEED,
        )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(tokenized_datasets['test'])
    preds = np.argmax(predictions.predictions, axis=-1)

    return preds, predictions.label_ids, metric.compute(predictions=preds, references=predictions.label_ids)


if __name__ == '__main__':
    mode = 'fine-tuning'
    # mode = 'p-tuning'

    # personality = ['A', 'C', 'E', 'O', 'N']
    personality = ['A']
    results = {}
    for p in personality:
        # data = '../data/Friends_'+p+'.tsv'
        data = '../data/Friends_'+p+'_with_role.tsv'

        preds, labels, f1 = training(data, mode)
        results[p] = {
            'preds': preds,
            'labels': labels,
            'f1': f1
        }
    for k, v in results.items():
        print('Personality', k, ':')
        for k_, v_ in v.items():
            print(k_, v_)
        print('------\n')








