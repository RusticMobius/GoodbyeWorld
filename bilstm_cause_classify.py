# -*- coding: UTF-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLstm(nn.Module):
    def __init__(self,
                 vocab_size,        #词表大小
                 embedding_dimension,   #词向量维度
                 dropout,       #随即失活
                 class_num,     #类别数
                 vectors,
                 hidden_size = 128,     #lstm隐藏层
                 hidden_size2 = 64,
                 num_layers = 2,        #lstm层数
                 ):
        super(BiLstm, self).__init__()
        #  self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=vocab_size - 1)
        self.lstm = nn.LSTM(embedding_dimension, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, class_num)

    def forward(self, x):
        x = self.embedding(x)  # #经过嵌入层之后x的维度，[句子长度,一个batch中所包含的样本数,词向量维度] 例：[3451,128,300]
        emb = x.permute(1, 0, 2)  # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        H,_ = self.lstm(emb)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        return out



