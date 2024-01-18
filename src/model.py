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
        self.num_classes = args.num_class
        self.device      = args.device
        self.hidden      = 32

        self.position_embedding     = Positional_Encoding(embed=7, pad_size=self.pad_size, device=self.device)
        self.semantic_encoder       = Encoder(dim_model=7, num_head=self.num_head, hidden=self.hidden)
        self.fc1 = nn.Linear(7, self.num_classes)
        
    def forward(self, emo_embedding, dialog_states, args):

        emo_embedding = self.position_embedding(emo_embedding) 
        emo_embedding = self.semantic_encoder(emo_embedding, dialog_states)

        zero           = torch.zeros_like(dialog_states)
        dialog_states  = torch.where(dialog_states<0, zero, dialog_states)

        speaker_length = torch.sum(dialog_states, dim=1)
        dialog_states  = torch.div(dialog_states, speaker_length.unsqueeze(1))
        
#         print(dialog_states)

        emo_logits   = torch.mul(emo_embedding, dialog_states.unsqueeze(2))
        emo_logits   = torch.sum(emo_logits, dim=1)

        emo_logits = self.fc1(emo_logits)
        return emo_logits


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

class HADE(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args            = args
        self.num_labels      = args.num_class
        self.config           = config
        self.roberta         = RobertaModel(config, add_pooling_layer=True)
        
        self.uttr_cls = nn.Linear(config.hidden_size, 2)

        self.emotion_encoder = Context_Encoder(args)
        self.sm = nn.Softmax(dim=1)

        self.init_weights()
    
    def forward(self, 
                uttr, uttr_mask, 
                personality_scores_vad,
                dialog_state, emo_embedding):  
        
        # print(emo_distribution.shape) # 16 * 20 * 7

        uttr_outputs  = self.roberta(uttr, uttr_mask)[1] 


        uttr_logits = self.uttr_cls(uttr_outputs)
        vad_logits = personality_scores_vad
        emo_logits = self.emotion_encoder(emo_embedding, dialog_state, self.args) # [batch_size * 2]
        
        # print(uttr_logits.shape)
        # print(vad_logits.shape)
        # print(emo_logits.shape)

        logits = self.sm(uttr_logits)  + self.sm(vad_logits) + self.sm(emo_logits)
#         logits = self.sm(emo_logits)# self.sm(uttr_logits)
        return logits