import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)

import evaluate
import torch

import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value

from datasets import load_metric
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup



model_name_or_path = "huggyllama/llama-7b"
task = "mrpc"
SEED = 42

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



input_sent = 'affective_sent'
tsv_file = '../new_data/CPED_A_with_role_1.tsv'
dataset = load_data(tsv_file, input_sent)

metric     = evaluate.load('accuracy')

import numpy as np



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
            "lr": 1e-4,
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples[input_sent], examples["personality_description"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[input_sent, "personality_description"],
)

# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")




peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="llama",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    deepspeed=deepspeed_config,
    load_best_model_at_end=True,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


predictions = trainer.predict(tokenized_datasets['test'])
preds = np.argmax(predictions.predictions, axis=-1)
print(metric.compute(predictions=preds, references=predictions.label_ids))







