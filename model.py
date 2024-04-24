import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model,dropout=0.1):
        super(Embedding, self).__init__()
        self.token = nn.Embedding(vocab_size,d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,sequencee):
        x = self.token(sequencee)
        return self.dropout(x)

def scaled_dot_product_attention(q, k, v, mask, adj):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores += (mask * -1e9)
    if adj is not None:
        scores += adj
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask, adj):

        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linear_layers, (q, k, v))]
        x = scaled_dot_product_attention(q, k, v, mask, adj)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)

        return self.output_linear(x)


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.))))


class Point_Wise_Feed_Forward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(Point_Wise_Feed_Forward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(gelu(self.w_1(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_head)
        self.ffn = Point_Wise_Feed_Forward(d_model, d_ff)
        self.ln = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, adj):
        
        mask = mask[:,None,None,:].to(torch.float32)
        attn_output = self.mha(x, x, x, mask, adj)
        attn_output = self.dropout(attn_output)
        out1 = self.ln(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out2 = self.ln(out1 + ffn_output)
        return out2


class Encoder_GNN(nn.Module):
    def __init__(self, cfg):
        super(Encoder_GNN, self).__init__()


        num_head = cfg.MODEL.NUM_HEAD
        d_model = num_head * 57
        d_ff = cfg.MODEL.D_FF
        num_layer = cfg.MODEL.NUM_LAYER

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_head, d_ff) for _ in range(num_layer)])
        self.num_layer = num_layer
        self.d_model = d_model
        self.nh = num_head
        self.ln = nn.Linear(57,d_model)
        self.dropout = nn.Dropout(p=cfg.MODEL.DROPOUT)

    def forward(self,x, mask, adj):
        x = x.to(torch.float32)
        adj = adj[:,None,:,:]
        #x_list = []
        #for i in range(self.nh):
            #x_list.append(x)
        #x = torch.cat(x_list,dim=2)
        x = self.ln(x)
        for i in range(self.num_layer):
            x = self.enc_layers[i](x,mask,adj)
        return x

class BertModel_GNN(nn.Module):
    def __init__(self,cfg,encoder):
        super(BertModel_GNN, self).__init__()
        num_head = cfg.MODEL.NUM_HEAD
        d_model = num_head * 57
        dropout = cfg.MODEL.DROPOUT
        self.encoder = encoder
        self.fc1 = nn.Linear(d_model,d_model)
        self.activate = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_model,167)

    def forward(self,x,mask,adj):
        x = self.encoder(x,mask,adj)
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activate(x)
        x = self.dropout(x)
        out = self.fc2(x)
        #x2 = self.encoder(x2, adj)
        #x2 = x2[:, 0, :]
        #x2 = self.fc1(x2)
        #x2 = self.activate(x2)
        #x2 = self.dropout(x2)
        #out2 = self.fc2(x2)
        return out

class BertModel_HM(nn.Module):
    def __init__(self,cfg,encoder):
        super(BertModel_HM, self).__init__()
        vocab_size = cfg.DATA.VOCAB_SIZE + 4
        num_head = cfg.MODEL.NUM_HEAD
        d_model = num_head * 57
        
        #self.embed = Embedding(vocab_size, 57)
        #self.dropout = nn.Dropout(p=cfg.MODEL.DROPOUT)
        self.encoder = encoder
        self.fc1 = nn.Linear(d_model,d_model)
        self.ln = LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model,vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x,mask,adjoin_matrix):
        #x = self.embed(x)
        #x = self.dropout(x)
        x = self.encoder(x,mask,adjoin_matrix)
        x = self.fc1(x)
        x = gelu(x)
        x = self.ln(x)
        x = self.fc2(x)
        return self.softmax(x)

class BertModel(nn.Module):
    def __init__(self,cfg,encoder):
        super(BertModel, self).__init__()
        self.model_gnn = BertModel_GNN(cfg,encoder)
        #self.model_hm = BertModel_HM(cfg,encoder)

    def forward(self,x,mask,adj):
        out1 = self.model_gnn(x,mask,adj)
        #out2 = self.model_hm(x2,mask,adj)
        return out1

class PredictModel(nn.Module):
    def __init__(self,cfg,encoder):
        super(PredictModel, self).__init__()
        #vocab_size = cfg.DATA.VOCAB_SIZE + 4
        num_head = cfg.MODEL.NUM_HEAD
        d_model = num_head * 57
        num_task = cfg.MODEL.NUM_TASK
        dropout = cfg.MODEL.DROPOUT
        if cfg.DATA.TASK_TYPE == 'classification':
            out_dim = num_task * 2
        else:
            out_dim = num_task
        #self.embed = Embedding(vocab_size, 57)
        self.encoder = encoder
        self.fc1 = nn.Linear(d_model,d_model)
        self.activate = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_model,out_dim)

    def forward(self,x,mask,adj):
        x = self.encoder(x,mask,adj)
        x = x[:, 0, :]
        #x2 = self.embed(x2)
        #x2 = self.encoder(x2,adj)
        #x2 = x2[:, 0, :]
        #x = torch.cat((x1,x2),1)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
