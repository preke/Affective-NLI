import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


import warnings

warnings.filterwarnings("ignore")

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

from peft import (
    get_peft_config,  
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)

from peft import PeftModel, PeftConfig, LoraConfig, TaskType


model_name_or_path = "huggyllama/llama-7b" # "THUDM/chatglm-6b" # "huggyllama/llama-7b"
# model_name_or_path = "roberta-large"

tokenizer  = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
SEED       = 42
batch_size = 32
num_epochs = 3
metric     = evaluate.load('accuracy')




deepspeed_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "allgather_bucket_size": 1e6,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 1000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}



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
    return tokenizer(example["sent"], truncation=True, max_length=450)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def training(data, mode):
    dataset_dict = load_data(data)
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=["sent"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    if mode == 'fine-tuning':
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

        training_args = TrainingArguments(
            'fine-tuning',
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            save_strategy='epoch',

            num_train_epochs=num_epochs,
            learning_rate=2e-05,
            logging_strategy="epoch",
            evaluation_strategy='epoch',

            load_best_model_at_end=True,
            seed=SEED,
        )


    elif mode == 'p-tuning':

        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=2)

        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=20,
            encoder_hidden_size=128
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir="p-tuning",
            learning_rate=1e-3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            #             load_best_model_at_end=True,
            seed=SEED,
        )
    
    elif mode == 'lora':
        
        peft_config = LoraConfig(
            task_type="SEQ_CLS", inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
        )


        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=2)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        training_args = TrainingArguments(
            output_dir="lora",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            deepspeed=deepspeed_config,
            load_best_model_at_end=True,
        )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        #         callbacks=[CustomTrainingCallback],
    )

    # trainer.train()

    predictions = trainer.predict(tokenized_datasets['test'])
    preds = np.argmax(predictions.predictions, axis=-1)

    return preds, predictions.label_ids, metric.compute(predictions=preds, references=predictions.label_ids)


if __name__ == '__main__':
    for flow_len in [1]#, 0.25, 0.5, 0.75]:  # 0.25,
        for mode in ['lora']:  # , 'p-tuning' 'fine-tuning'
            start_time = time.time()
            personality = ['A', 'C', 'E', 'O', 'N']
            results = {}
            for p in personality:
                data = '../new_data/CPED_' + p + '_with_role_' + str(flow_len) + '.tsv'

                preds, labels, acc = training(data, mode)
                results[p] = {
                    'preds': list(preds),
                    'labels': list(labels),
                    'acc': acc
                }
            end_time = time.time()

            json_str = json.dumps(results, default=str)
            with open('results/CPED_origin_sent_' + str(flow_len) + '_' + mode + '_3_epoches.json', 'w') as json_file:
                json_file.write(json_str)

            print('Processing time', end_time - start_time, 's.')