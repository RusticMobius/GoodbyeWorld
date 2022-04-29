# -*- coding: UTF-8 -*-
stopwords_path = 'stopwords.txt'
stopwords = open(stopwords_path).read().split('\n')

import jieba


def cut(sentence):
    return [token for token in jieba.lcut(sentence) if token not in stopwords]


import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator,Field,TabularDataset,LabelField

TEXT = Field(sequential=True, lower=True, tokenize=cut)
LABEL = LabelField(sequential=False,use_vocab=True,is_target=True)

field = [('id',None),('label',LABEL),('content',TEXT)]
# 这里主要是告诉torchtext需要处理哪些数据，这些数据存放在哪里，TabularDataset是一个处理scv/tsv的常用类
train_data = TabularDataset(
    path="./data2/train_data.tsv",
    format="tsv",
    skip_header=False,
    # csv_reader_params={'delimiter':'\t'},
    fields=field
)
dev_data = TabularDataset(
    path="./data2/dev_data.tsv",
    format="tsv",
    skip_header=False,
    # csv_reader_params={'delimiter':'\t'},
    fields=field
)
test_data = TabularDataset(
    path="./data2/test_data.tsv",
    format="tsv",
    skip_header=False,
    # csv_reader_params={'delimiter':'\t'},
    fields=field
)
# train_dataset, dev_dataset, test_dataset = torchtext.legacy.data.TabularDataset.splits(
#     path='./data2',  # 文件存放路径
#     format='tsv',  # 文件格式
#     skip_header=False,  # 是否跳过表头，我这里数据集中没有表头，所以不跳过
#     train='train_data.tsv',
#     validation='dev_data.tsv',
#     test='test_data.tsv',
#     fields=[('index', None),('label', LABEL), ('content', TEXT)]  # 定义数据对应的表头
#
# )


pretrained_name = 'sgns.renmin.bigram-char'
pretrained_path = '/Users/scarlett/Downloads'
vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=pretrained_path)

TEXT.build_vocab(train_data, dev_data,test_data, vectors=vectors)
LABEL.build_vocab(train_data, dev_data, test_data)
label_list = LABEL.vocab.itos

train_iter = BucketIterator(
    dataset=train_data,
    batch_size=32,
    sort_key=lambda x: len(x.content),
    sort_within_batch=True,
    device=torch.device('cpu')

)
dev_iter = BucketIterator(
    dataset=dev_data,
    batch_size=128,
    sort_key=lambda x: len(x.content),
    sort_within_batch=True,
    device=torch.device('cpu')

)
test_iter = BucketIterator(
    dataset=test_data,
    batch_size=1,
    sort_key=lambda x: len(x.content),
    sort_within_batch=True,
    device=torch.device('cpu')

)
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self,
                 class_num,  # 最后输出的种类数
                 filter_sizes,  # 卷积核的长也就是滑动窗口的长
                 filter_num,  # 卷积核的数量
                 vocabulary_size,  # 词表的大小
                 embedding_dimension,  # 词向量的维度
                 vectors,  # 词向量
                 dropout):  # dropout率
        super(TextCNN, self).__init__()  # 继承nn.Module

        chanel_num = 1  # 通道数，也就是一篇文章一个样本只相当于一个feature map

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)  # 嵌入层
        # self.embedding = self.embedding.from_pretrained(vectors)  # 嵌入层加载预训练词向量

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (fsz, embedding_dimension), padding=(2, 0)) for fsz in
             filter_sizes])  # 卷积层
        self.dropout = nn.Dropout(dropout)  # dropout
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)  # 全连接层

    def forward(self, x):
        # x维度[句子长度,一个batch中所包含的样本数] 例:[3451,128]
        x = self.embedding(x)  # #经过嵌入层之后x的维度，[句子长度,一个batch中所包含的样本数,词向量维度] 例：[3451,128,300]
        x = x.permute(1, 0, 2)  # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.unsqueeze(
            1)  # # conv2d需要输入的是一个四维数据，所以新增一维feature map数 unsqueeze(1)表示在第一维处新增一维，[一个batch中所包含的样本数,一个样本中的feature map数，句子长度,词向量维度] 例：[128,1,3451,300]
        x = [conv(x) for conv in
             self.convs]  # 与卷积核进行卷积，输出是[一个batch中所包含的样本数,卷积核数，句子长度-卷积核size+1,1]维数据,因为有[3,4,5]三张size类型的卷积核所以用列表表达式 例：[[128,16,3459,1],[128,16,3458,1],[128,16,3457,1]]
        x = [sub_x.squeeze(3) for sub_x in
             x]  # squeeze(3)判断第三维是否是1，如果是则压缩，如不是则保持原样 例：[[128,16,3459],[128,16,3458],[128,16,3457]]
        x = [F.relu(sub_x) for sub_x in x]  # ReLU激活函数激活，不改变x维度
        x = [F.max_pool1d(sub_x, sub_x.size(2)) for sub_x in
             x]  # 池化层，根据之前说的原理，max_pool1d要取出每一个滑动窗口生成的矩阵的最大值，因此在第二维上取最大值 例：[[128,16,1],[128,16,1],[128,16,1]]
        x = [sub_x.squeeze(2) for sub_x in x]  # 判断第二维是否为1，若是则压缩 例：[[128,16],[128,16],[128,16]]
        x = torch.cat(x, 1)  # 进行拼接，例：[128,48]
        x = self.dropout(x)
        logits = self.fc(x)  # 全接连层 例：输入x是[128,48] 输出logits是[128,10]
        return logits


class_num = len(LABEL.vocab)  # 类别数目
filter_size = [3, 4, 5]  # 卷积核种类数
filter_num = 16  # 卷积核数量
vocab_size = len(TEXT.vocab)  # 词表大小
embedding_dim = TEXT.vocab.vectors.size()[-1]  # 词向量维度
vectors = TEXT.vocab.vectors  # 词向量
dropout = 0.5
learning_rate = 0.001  # 学习率
epochs = 5  # 迭代次数
save_dir = './model1'  # 模型保存路径
steps_show = 1  # 每10步查看一次训练集loss和mini batch里的准确率
steps_eval = 32  # 每100步测试一下验证集的准确率
early_stopping = 2000  # 若发现当前验证集的准确率在1000步训练之后不再提高 一直小于best_acc,则提前停止训练

textcnn_model = TextCNN(class_num=class_num,
                        filter_sizes=filter_size,
                        filter_num=filter_num,
                        vocabulary_size=vocab_size,
                        embedding_dimension=embedding_dim,
                        vectors=vectors,
                        dropout=dropout)


def train(train_iter, dev_iter, model):
    if torch.cuda.is_available():  # 判断是否有GPU，如果有把模型放在GPU上训练，速度质的飞跃
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 梯度下降优化器，采用Adam
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        for batch in train_iter:
            feature, target = batch.content, batch.label
            if torch.cuda.is_available():  # 如果有GPU将特征更新放在GPU上
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()  # 将梯度初始化为0，每个batch都是独立训练地，因为每训练一个batch都需要将梯度归零
            logits = model(feature)
            loss = F.cross_entropy(logits, target)  # 计算损失函数 采用交叉熵损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 放在loss.backward()后进行参数的更新
            steps += 1
            if steps % steps_show == 0:  # 每训练多少步计算一次准确率，我这边是1，可以自己修改
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()  # logits是[128,10],torch.max(logits, 1)也就是选出第一维中概率最大的值，输出为[128,1],torch.max(logits, 1)[1]相当于把每一个样本的预测输出取出来，然后通过view(target.size())平铺成和target一样的size (128,),然后把与target中相同的求和，统计预测正确的数量
                train_acc = 100.0 * corrects / batch.batch_size  # 计算每个mini batch中的准确率
                print('steps:{} - loss: {:.6f}  acc:{:.4f}'.format(
                    steps,
                    loss.item(),
                    train_acc))

            if steps % steps_eval == 0:  # 每训练100步进行一次验证
                dev_acc = dev_eval(dev_iter, model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, save_dir, steps)
                else:
                    if steps - last_step >= early_stopping:
                        print('\n提前停止于 {} steps, acc: {:.4f}%'.format(last_step, best_acc))
                        raise KeyboardInterrupt

# textcnn_model.load_state_dict(torch.load('model1/bestmodel_steps288(b32).pt'))

def dev_eval(dev_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in dev_iter:
        feature, target = batch.content, batch.label
        #print(target)
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(dev_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

    return accuracy


import os


# 定义模型保存函数
def save(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = 'bestmodel_steps{}(b32).pt'.format(steps)
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model.state_dict(), save_bestmodel_path)

#train(train_iter, dev_iter, textcnn_model)
#dev_eval(dev_iter,textcnn_model)


import csv


def predict(model,info):
    model.eval()
    f = open("./data2/temp_input.tsv", 'w')
    writer = csv.writer(f, delimiter='\t')
    writer.writerow([0, '离婚纠纷', info])
    f.close()

    textcnn_model.eval()

    test_data = TabularDataset(
        path="./data2/temp_input.tsv",
        format="tsv",
        skip_header=False,
        fields=field
    )

    test_iter = BucketIterator(
        dataset=test_data,
        batch_size=1,
        sort_key=lambda x: len(x.content),
        sort_within_batch=True,
        device=torch.device('cpu')
    )

    label = "SOS"
    for batch in test_iter:
        # count += 1
        feature = batch.content
        logits = model(feature)
        #print(target)
        label = label_list[torch.max(logits, 1)[1].detach().numpy()[0]]

    return label



