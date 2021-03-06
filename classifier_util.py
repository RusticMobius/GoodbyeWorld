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
    class_num = len(LABEL.vocab)  # ????????????
    vocab_size = len(TEXT.vocab)  # ????????????
    embedding_dim = TEXT.vocab.vectors.size()[-1]  # ???????????????
    vectors = TEXT.vocab.vectors  # ?????????
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
steps_show = 1  # ???10????????????????????????loss???mini batch???????????????
steps_eval = 100  # ???100????????????????????????????????????
early_stopping = 2000  # ???????????????????????????????????????1000??????????????????????????? ????????????best_acc,?????????????????????
save_dir = './model'  # ??????????????????

def train(model, train_iter, dev_iter):
    best_acc = 0
    last_step = 0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = 0  # ?????????????????????batch
    dev_best_loss = float('inf')
    last_improve = 0  # ?????????????????????loss?????????batch???
    flag = False  # ????????????????????????????????????
    for epoch in range(epochs_num):
        print('Epoch [{}/{}]'.format(epoch + 1, epochs_num))
        for batch in train_iter:
            feature, target = batch.content, batch.label
            if torch.cuda.is_available():  # ?????????GPU?????????????????????GPU???
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()  # ?????????????????????0?????????batch?????????????????????????????????????????????batch????????????????????????
            logits = model(feature)
            loss = F.cross_entropy(logits, target)  # ?????????????????? ???????????????????????????
            loss.backward()  # ????????????
            optimizer.step()  # ??????loss.backward()????????????????????????
            steps += 1
            if steps % steps_show == 0:
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()  # logits???[128,10],torch.max(logits, 1)?????????????????????????????????????????????????????????[128,1],torch.max(logits, 1)[1]??????????????????????????????????????????????????????????????????view(target.size())????????????target?????????size (128,),????????????target????????????????????????????????????????????????
                train_acc = 100.0 * corrects / batch.batch_size  # ????????????mini batch???????????????
                print('steps:{} - loss: {:.6f}  acc:{:.4f}'.format(
                    steps,
                    loss.item(),
                    train_acc))
            if steps % steps_eval == 0:  # ?????????100?????????????????????
                dev_acc = dev_eval(dev_iter, model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, save_dir, steps)
                else:
                    if steps - last_step >= early_stopping:
                        print('\n??????????????? {} steps, acc: {:.4f}%'.format(last_step, best_acc))
                        raise KeyboardInterrupt

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def dev_eval(dev_iter, model):
    with torch.no_grad():
        model.eval()
        corrects, avg_loss = 0, 0
        preds_list = []
        target_list = []
        for batch in dev_iter:
            feature, target = batch.content, batch.label
            # print(target)
            if torch.cuda.is_available():
                feature, target = feature.cuda(), target.cuda()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            preds_list.extend(torch.max(logits, 1)
                         [1].detach().cpu().numpy())
            target_list.extend(target.detach().cpu().numpy())

            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)
                         [1].view(target.size()).data == target.data).sum()
        size = len(dev_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        sklearn_f1 = f1_score(target_list, preds_list, average='micro')
        sklearn_precision = precision_score(target_list, preds_list, average='micro')
        sklearn_recall = recall_score(target_list, preds_list, average='micro')
        print(classification_report(target_list, preds_list))
        conf_matrix = get_confusion_matrix(target_list, preds_list)
        plot_confusion_matrix(conf_matrix)
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))

        return accuracy

def get_confusion_matrix(true, pred):
    label = [i for i in range(13)]
    conf_matrix = confusion_matrix(true, pred, labels=label)
    return conf_matrix

def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    indices = range(conf_matrix.shape[0])
    label = [i for i in range(13)]
    plt.xticks(indices, label)
    plt.yticks(indices, label)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


import os
# ????????????????????????
def save(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = 'TEXTCNN_model_steps{}(b64).pt'.format(steps)
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model.state_dict(), save_bestmodel_path)


import csv


def predict(model, info):
    with torch.no_grad():
        model.eval()
        f = open("./data2/temp_input.tsv", 'w')
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([0, '????????????', info])
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
#bilstm_model.load_state_dict(torch.load('model2/bilstmAtt_model_steps448(b32).pt'))
#textcnn_model.load_state_dict(torch.load('model/bestmodel_steps200.pt'))
dpcnn_model.load_state_dict(torch.load('model3/DPCNN_model_steps300(b64).pt'))
#train(textcnn_model,train_iter,dev_iter)
dev_eval(dev_iter,dpcnn_model)