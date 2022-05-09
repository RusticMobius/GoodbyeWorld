import jieba
import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator, Field, TabularDataset, LabelField
import torch.nn.functional as F


def load_stopwords():
    stopwords1 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
    stopwords2 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
    stopwords3 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
    stopwords4 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
    stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4
    return stopwords


def cut(sentence):
    return [token for token in jieba.lcut(sentence) if token not in stopwords]


stopwords = load_stopwords()
TEXT = Field(sequential=True, lower=True, tokenize=cut)
LABEL = LabelField(sequential=False, use_vocab=True, is_target=True)
field = [('id', None), ('label', LABEL), ('content', TEXT)]

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

pretrained_name = 'sgns.renmin.bigram-char'
pretrained_path = '/Users/scarlett/Downloads/'
vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=pretrained_path)

TEXT.build_vocab(train_data, dev_data, test_data, vectors=vectors)
LABEL.build_vocab(train_data, dev_data, test_data)
label_list = LABEL.vocab.itos

class Config(object):
    class_num = len(LABEL.vocab)  # 类别数目
    vocab_size = len(TEXT.vocab)  # 词表大小
    embedding_dim = TEXT.vocab.vectors.size()[-1]  # 词向量维度
    vectors = TEXT.vocab.vectors  # 词向量
    dropout = 0.5
    learning_rate = 1e-3


train_iter = BucketIterator(
    dataset=train_data,
    batch_size=64,
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

learning_rate = 1e-3
dropout = 0.5
epochs_num = 10
steps_show = 1  # 每10步查看一次训练集loss和mini batch里的准确率
steps_eval = 100  # 每100步测试一下验证集的准确率
early_stopping = 2000  # 若发现当前验证集的准确率在1000步训练之后不再提高 一直小于best_acc,则提前停止训练
save_dir = './model3'  # 模型保存路径

def train(model, train_iter, dev_iter):
    best_acc = 0
    last_step = 0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(epochs_num):
        print('Epoch [{}/{}]'.format(epoch + 1, epochs_num))
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
            if steps % steps_show == 0:
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

from sklearn.metrics import classification_report

def dev_eval(dev_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in dev_iter:
        feature, target = batch.content, batch.label
        # print(target)
        if torch.cuda.is_available():
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        print(classification_report(logits,target))
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
    save_path = 'DPCNN_model_steps{}(b64).pt'.format(steps)
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model.state_dict(), save_bestmodel_path)


import csv


def predict(model, info):
    model.eval()
    f = open("./data2/temp_input.tsv", 'w')
    writer = csv.writer(f, delimiter='\t')
    writer.writerow([0, '离婚纠纷', info])
    f.close()

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
        # print(target)
        label = label_list[torch.max(logits, 1)[1].detach().numpy()[0]]

    return label

from bilstm_cause_classify import BiLstm

config = Config()

bilstm_model = BiLstm(
    class_num=config.class_num,
    vocab_size=config.vocab_size,
    embedding_dimension=config.embedding_dim,
    dropout=config.dropout,
    vectors=config.vectors
)

from textcnnClassify import TextCNN

textcnn_model = TextCNN(class_num=config.class_num,
                        vocabulary_size=config.vocab_size,
                        embedding_dimension=config.embedding_dim,
                        vectors=config.vectors,
                        dropout=config.dropout)

from dpcnn_cause_classify import DPCNN

dpcnn_model = DPCNN(class_num=config.class_num,
                    vocab_size=config.vocab_size,
                    vectors=config.vectors,
                    embedding_dimension=config.embedding_dim,
                    dropout=config.dropout
)

dpcnn_model.load_state_dict(torch.load('model3/DPCNN_model_steps300(b64).pt'))
# train(dpcnn_model,train_iter,dev_iter)
dev_eval(dev_iter,dpcnn_model)