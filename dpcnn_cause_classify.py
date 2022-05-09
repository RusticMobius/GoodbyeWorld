import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DPCNN(nn.Module):
    def __init__(self,
                 vocab_size,  # 词表大小
                 embedding_dimension,  # 词向量维度
                 dropout,  # 随机失活
                 class_num,  # 类别数
                 vectors,
                 filter_num=250  # 卷积核数量(channels数)
                 ):
        super(DPCNN,self).__init__()
        #  self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, vocab_size -1)

        self.conv_region = nn.Conv2d(1, filter_num, (3, embedding_dimension), stride=1)
        self.conv = nn.Conv2d(filter_num, filter_num, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(filter_num, class_num)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        #print(x.shape)
        while x.size()[2] > 1:
            x = self._block(x)
            #print(x.shape)

        x = x.squeeze()
        # print(x.shape)
        # x = x.view(x.size(0),-1)
        #print("1",x.shape)
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

