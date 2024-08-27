import pandas as pd
df_peld = pd.read_csv('../data/Dyadic_PELD.tsv', sep='\t')
df_Emotion = pd.DataFrame([], columns=['Utterance', 'Emotion', 'Sentiment'])
df_Emotion['Utterance'] = list(df_peld['Utterance_1']) + list(df_peld['Utterance_2']) + list(df_peld['Utterance_3'])
df_Emotion['Emotion'] = list(df_peld['Emotion_1']) + list(df_peld['Emotion_2']) + list(df_peld['Emotion_3'])
df_Emotion['Sentiment'] = list(df_peld['Sentiment_1']) + list(df_peld['Sentiment_2']) + list(df_peld['Sentiment_3'])
df_Emotion.head()

# ========

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

MAX_LEN =256
SEED = 0
batch_size = 16


labels = df_Emotion['Emotion']
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
label_enc    = labelencoder.fit_transform(labels)
labels       = label_enc


# tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
# model     = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7).cuda(1)


from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")

model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base", num_labels=7).cuda(1)


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
# model     = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7).cuda(1)

input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN,pad_to_max_length=True) for sent in df_Emotion['Utterance']]

attention_masks = []
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

train_inputs,test_inputs,train_labels,test_labels = \
    train_test_split(input_ids, labels, random_state=SEED, test_size=0.1, stratify=labels)

train_masks,test_masks,_,_ = train_test_split(attention_masks,labels,random_state=SEED,test_size=0.1,  stratify=labels)

train_inputs      = torch.tensor(train_inputs)
test_inputs       = torch.tensor(test_inputs)
train_labels        = torch.tensor(train_labels)    
test_labels         = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
test_masks = torch.tensor(test_masks)


train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

train_length = len(train_data)


# ========


import numpy as np

# Parameters:
lr = 1e-6
adam_epsilon = 1e-8
epochs = 15

num_warmup_steps = 0
num_training_steps = len(train_dataloader)*epochs

optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon,correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,matthews_corrcoef
from tqdm import tqdm, trange,tnrange,tqdm_notebook

## Store our loss and accuracy for plotting
train_loss_set = []
learning_rate = []

# Gradients gets accumulated by default
model.zero_grad()

for _ in tnrange(1,epochs+1,desc='Epoch'):
    print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
    # Calculate total loss for this epoch
    batch_loss = 0

    for step, batch in enumerate(train_dataloader):
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        # Add batch to GPU
        batch = tuple(t.cuda(1) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        
        # Backward pass
        loss.backward()
        
        # Clip the norm of the gradients to 1.0
        # Gradient clipping is not in AdamW anymore
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        
        # Update learning rate schedule
        scheduler.step()

        # Clear the previous accumulated gradients
        optimizer.zero_grad()
        
        # Update tracking variables
        batch_loss += loss.item()

    # Calculate the average loss over the training data.
    avg_train_loss = batch_loss / len(train_dataloader)

    #store the current learning rate
    for param_group in optimizer.param_groups:
        print("\n\tCurrent Learning rate: ",param_group['lr'])
        learning_rate.append(param_group['lr'])
      
    train_loss_set.append(avg_train_loss)
    print(F'\n\tAverage Training loss: {avg_train_loss}')
      
    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables 
    eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0
    
    labels_list = np.array([])
    pred_list = np.array([])

    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda(1) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          logits = model(b_input_ids, attention_mask=b_input_mask)
        
        # Move logits and labels to CPU
        logits = logits[0].to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        pred_list = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)
        
        df_metrics=pd.DataFrame({'Epoch':epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})
        
        tmp_eval_accuracy = accuracy_score(labels_flat,pred_flat)
        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)
        
        eval_accuracy += tmp_eval_accuracy
        eval_mcc_accuracy += tmp_eval_mcc_accuracy
        nb_eval_steps += 1

    print(classification_report(pred_list, labels_list, digits=4))