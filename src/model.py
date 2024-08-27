from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import TransformerEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head)

    def forward(self, x, dialog_states):
        out = self.attention(x, dialog_states)
        return out

class Dialog_State_Encoding(nn.Module):
    def __init__(self, embed, pad_size, device):
        super(Dialog_State_Encoding, self).__init__()
        self.device = device
        self.dim_model = embed # 32
        self.pad_size = pad_size # 30

    
    def forward(self, x, dialog_states):
        dialog_states = dialog_states.float().unsqueeze(-1).expand(-1,-1,self.dim_model)        
        out = x + nn.Parameter(dialog_states, requires_grad=False).to(self.device)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        return out


class Context_Encoder(nn.Module):
    def __init__(self, args):
        super(Context_Encoder, self).__init__()

        self.args        = args
        self.pad_size    = args.MAX_NUM_UTTR
        self.num_head    = 1
        self.dim_model   = args.d_transformer
        self.num_encoder = 1
        self.num_classes = args.num_class
        self.device      = args.device
        self.hidden      = 512 # 256

        self.position_embedding     = Positional_Encoding(embed=self.dim_model, pad_size=self.pad_size, device=self.device)
        self.dialog_state_embedding = Dialog_State_Encoding(embed=self.dim_model, pad_size=self.pad_size, device=self.device)
        self.semantic_encoder       = Encoder(dim_model=self.dim_model, num_head=self.num_head, hidden=self.hidden)
        if args.mode == 'Context_Hierarchical_affective':
            self.affective_encoder      = Encoder(dim_model=self.dim_model, num_head=self.num_head, hidden=self.hidden)
            self.fc1 = nn.Linear(self.dim_model*2, self.num_classes)
        else:
            self.fc1 = nn.Linear(self.dim_model, self.num_classes)
        
    def forward(self, x, dialog_states, context_vad, d_transformer, args):
        # Semantic Aspect:
        semantic_out   = x.view(-1, self.pad_size, self.dim_model)
        semantic_out   = self.position_embedding(semantic_out) 
        semantic_out   = self.semantic_encoder(semantic_out, dialog_states)

        if args.mode == 'Context_Hierarchical_affective':
            # Affective aspect:
            affective_out  = x.view(-1, self.pad_size, self.dim_model)
            affective_out  = self.position_embedding(affective_out)
            affective_out  = self.affective_encoder(affective_out, dialog_states)

        zero           = torch.zeros_like(dialog_states)
        dialog_states  = torch.where(dialog_states<0, zero, dialog_states)
        speaker_length = torch.sum(dialog_states, dim=1)
        dialog_states  = torch.div(dialog_states, speaker_length.unsqueeze(1))
        
        semantic_out   = torch.mul(semantic_out, dialog_states.unsqueeze(2))
        semantic_out   = torch.sum(semantic_out, dim=1)
        if args.mode == 'Context_Hierarchical_affective':
            # affective_out  = torch.mul(affective_out, dialog_states.unsqueeze(2))
            # affective_out  = torch.sum(affective_out, dim=1)
            affective_out  = torch.mean(affective_out, dim=1)
        
            out = torch.cat([semantic_out, affective_out], dim=1)
        else:
            out = semantic_out
        out = self.fc1(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale, mask):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        
        mask = mask.unsqueeze(-1).expand(-1, -1, attention.shape[1])
        mask = mask + 1
        attention = attention * scale
        attention = attention.masked_fill_(mask == 0, -1e9)


        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, dialog_states):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # print(Q.shape, K.shape, V.shape)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale, dialog_states)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = context + x  # 残差连接
        out = self.layer_norm(out)
        return out



class DialogVAD(BertPreTrainedModel):
    
    def __init__(self, config, args):
        super().__init__(config)
        self.args            = args
        self.num_labels      = args.num_class
        self.d_transformer   = args.d_transformer
        self.config          = config
        self.bert            = BertModel(config)
        self.reduce_size     = nn.Linear(config.hidden_size, self.d_transformer) 
        self.vad_to_hidden = nn.Linear(3, self.d_transformer)

        self.context_encoder = Context_Encoder(args)
        self.emo_cls     = nn.Linear(config.hidden_size, 7) 
        self.init_weights()
    
    def forward(self, context, context_mask, dialog_states, context_vad):  
        
        batch_size, max_ctx_len, max_utt_len = context.size() # 16 * 30 * 32
        
        context_utts = context.view(max_ctx_len, batch_size, max_utt_len)    
        context_mask = context_mask.view(max_ctx_len, batch_size, max_utt_len)   

        uttr_outputs  = [self.bert(uttr, uttr_mask) for uttr, uttr_mask in zip(context_utts,context_mask)]

        uttr_outputs  = [self.reduce_size(uttr_output[1]) for uttr_output in uttr_outputs] 
        uttr_embeddings = torch.stack(uttr_outputs) # 30 * 16 * 768
        uttr_embeddings = torch.autograd.Variable(uttr_embeddings.view(batch_size, max_ctx_len, self.d_transformer), requires_grad=True)

        if self.args.mode == 'Context_Hierarchical_affective':
            context_vad = context_vad.view(max_ctx_len, batch_size, 3)
            context_vad = [self.vad_to_hidden(uttr_vad) for uttr_vad in context_vad]
            context_vad = torch.stack(context_vad)
            context_vad = torch.autograd.Variable(context_vad.view(batch_size, max_ctx_len, self.d_transformer), requires_grad=True)
        else:
            context_vad = 0
        
        # ---- concat with transformer to do the self-attention
        logits = self.context_encoder(uttr_embeddings, dialog_states, context_vad, self.d_transformer, self.args) # [batch_size * 2]

        return logits



class DialogVAD_roberta(RobertaPreTrainedModel):
    
    def __init__(self, config, args):
        super().__init__(config)
        self.args            = args
        self.num_labels      = args.num_class
        self.d_transformer   = args.d_transformer
        self.config          = config
        self.roberta         = RobertaModel(config, add_pooling_layer=True)
        self.reduce_size     = nn.Linear(config.hidden_size, self.d_transformer) 
        self.vad_to_hidden = nn.Linear(3, self.d_transformer)

        self.context_encoder = Context_Encoder(args)
        self.emo_cls     = nn.Linear(config.hidden_size, 7) 
        self.init_weights()
    
    def forward(self, context, context_mask, dialog_states, context_vad):  
        
        batch_size, max_ctx_len, max_utt_len = context.size() # 16 * 30 * 32
        

        context_utts = context.view(max_ctx_len, batch_size, max_utt_len)    
        context_mask = context_mask.view(max_ctx_len, batch_size, max_utt_len)   

        uttr_outputs  = [self.roberta(uttr, uttr_mask) for uttr, uttr_mask in zip(context_utts,context_mask)]


        uttr_outputs  = [self.reduce_size(uttr_output[1]) for uttr_output in uttr_outputs] 
        uttr_embeddings = torch.stack(uttr_outputs) # 30 * 16 * 768
        uttr_embeddings = torch.autograd.Variable(uttr_embeddings.view(batch_size, max_ctx_len, self.d_transformer), requires_grad=True)

        if self.args.mode == 'Context_Hierarchical_affective':
            context_vad = context_vad.view(max_ctx_len, batch_size, 3)
            context_vad = [self.vad_to_hidden(uttr_vad) for uttr_vad in context_vad]
            context_vad = torch.stack(context_vad)
            context_vad = torch.autograd.Variable(context_vad.view(batch_size, max_ctx_len, self.d_transformer), requires_grad=True)
        else:
            context_vad = 0
        
        


        # ---- concat with transformer to do the self-attention
        logits = self.context_encoder(uttr_embeddings, dialog_states, context_vad, self.d_transformer, self.args) # [batch_size * 2]

        return logits

    