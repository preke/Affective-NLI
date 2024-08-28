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



model_name_or_path = "meta-llama/Llama-2-7b-chat-hf" #"huggyllama/llama-7b"
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
tsv_file = '../data/Friends_A_with_role_1.tsv'
dataset = load_data(tsv_file, input_sent)

metric     = evaluate.load('accuracy')

import numpy as np





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
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
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







