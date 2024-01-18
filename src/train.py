import sklearn
import pandas as pd
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm, trange,tnrange, tqdm_notebook
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import traceback
import os
import shutil



def train_model(model, args, train_dataloader, valid_dataloader, train_length):
    if args.data == 'Friends_Persona':
        num_warmup_steps = int(0*train_length)
    else:
        num_warmup_steps   = int(0.05*train_length)
    num_training_steps = len(train_dataloader)*args.epochs

    if args.BASE == 'RoBERTa':
        base_encoding_layer = 'roberta'
        
    if args.mode == 'HADE':
        pass
#         for name, param in model.named_parameters():        
#             if name.startswith(base_encoding_layer):
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
#                 print(name,param.size())
    
   
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    
    train_loss_set = []
    learning_rate  = []
    model.zero_grad()
    best_eval_f1 = 0

    for _ in tnrange(1, args.epochs+1, desc='Epoch'):
        print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
        # Calculate total loss for this epoch
        batch_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()
            
            # Add batch to GPU
            batch = tuple(t.cuda(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            if args.mode == 'HADE':

                b_uttr, b_uttr_mask, b_vad_personality, b_emotions, b_dialog_states, b_labels = batch

                logits = model(b_uttr, b_uttr_mask, b_vad_personality, b_dialog_states, b_emotions)
                
                loss_ce             = nn.CrossEntropyLoss()
                classification_loss = loss_ce(logits, b_labels)
                
                loss                = classification_loss


            elif args.mode == 'Uttr' or args.mode == 'Full_dialog':
                b_input_ids, b_input_mask, b_labels = batch
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss    = outputs.loss
                logits  = outputs.logits
            elif args.mode == 'Context':
                b_input_ids, b_input_mask, b_vad_scores, b_labels = batch
                b_seg_embeddings = b_vad_scores
                # outputs = model(b_input_ids, token_type_ids=b_seg_embeddings, attention_mask=b_input_mask, labels=b_labels)
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss    = outputs.loss
                logits  = outputs.logits
            
            # Backward pass
            loss.backward()
            
            # Clip the norm of the gradients to 1.0
            # Gradient clipping is not in AdamW anymore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update tracking variables
            batch_loss += loss.item()


            # Validation
            # Put model in evaluation mode to evaluate loss on the validation set
            if step % 1 == 0:
                args.print_eval = False
                eval_acc, eval_f1 = eval_model(model, args, valid_dataloader)
                if eval_f1 > best_eval_f1:
                            best_eval_f1 = eval_f1
                            try:
                                shutil.rmtree(args.model_path)
                            except:
                                print(traceback.print_exc())
                                os.mkdir(args.model_path)
                            try:
                                model.save_pretrained(args.model_path)
                                print('****** saved new model to ' + args.model_path + ' ******')
                            except:
                                print(traceback.print_exc())
                else:
                    pass


        # Calculate the average loss over the training data.
        avg_train_loss = batch_loss / len(train_dataloader)
        train_loss_set.append(avg_train_loss)

    return train_loss_set, best_eval_f1

    
    
    

def eval_model(model, args, valid_dataloader):
    # Tracking variables 
    eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0, 0, 0
    
    labels_list = np.array([])
    pred_list = np.array([])

    # Evaluate data for one epoch
    for batch in valid_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda(args.device) for t in batch)
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        model.eval()
        with torch.no_grad():
            if args.mode == 'HADE':
                

                b_uttr, b_uttr_mask, b_vad_personality, b_emotions, b_dialog_states, b_labels = batch

                logits = model(b_uttr, b_uttr_mask, b_vad_personality, b_dialog_states, b_emotions)
                
                loss_ce             = nn.CrossEntropyLoss()
                classification_loss = loss_ce(logits, b_labels)
                
                loss                = classification_loss
                
               
            elif args.mode == 'Uttr' or args.mode == 'Full_dialog':
                b_input_ids, b_input_mask, b_labels = batch
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                logits  = outputs.logits

            elif args.mode == 'Context':
                b_input_ids, b_input_mask, b_vad_scores, b_labels = batch    
                b_seg_embeddings = b_vad_scores ## only in this case
                # outputs = model(b_input_ids, token_type_ids=b_seg_embeddings, attention_mask=b_input_mask, labels=b_labels)
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss    = outputs.loss
                logits  = outputs.logits
                
        # Move logits and labels to CPU
        logits      = logits.to('cpu').numpy()

        
        label_ids   = b_labels.to('cpu').numpy()
        
        pred_flat   = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        pred_list   = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)
                
        nb_eval_steps += 1
    if args.print_eval == True:
        print('Predict:',pred_list)
        print('Labels:',labels_list)
    return accuracy_score(labels_list, pred_list), f1_score(labels_list, pred_list)





