# -*- coding: UTF-8 -*-
from causeClassify import textcnn_model,cut
import torchtext
import torch
import jieba
import csv

def cut1(sentence):
    stopwords1 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
    stopwords2 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
    stopwords3 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
    stopwords4 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
    stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4
    return [token for token in jieba.lcut(sentence) if token not in stopwords]

test_string = '原告诉称，原、被告确立恋爱关系后准备结婚，原告给付被告结婚彩礼30000元，另给付预定酒店、婚庆公司各项费用6600元，因在筹备婚礼过程中矛盾不断，且以被告与其他异性朋友谈恋爱为名多次到原告家中对原告之母进行辱骂，现原、被告经协商解除婚约。被告拒绝退还原告聘礼，诉请返还聘礼30000元及婚庆损失500元，负担诉讼费。 被告辩称，原告所述取款凭条是其母闫桂玲的名字，原告没给被告30000元，被告给付原告10000元购买衣物，如返还也应返还20000元，不同意原告诉请。 经审理查明，原、被告相识恋爱，期间原告将其母名下存折一个给付被告，被告从中支取30000元。原告为结婚筹备婚礼，向财富豪为酒店预付婚宴押金5600元，向金舰婚庆服务中心支付婚庆定金1000元，后双方因故发生矛盾，致协商解除婚约，但就退还聘礼未能达成一致意见，原告诉请返还。审理中通过找财富豪为酒店，该酒店返还原告押金，故原告撤销此2800元诉讼请求，诉请被告支付婚庆定金1000元的二分之一即500元及聘礼30000元。被告则以其给付原告10000元为抗辩理由，主张给付原告20000元，遭原告拒绝，调解未果。另查，原、被告未办理结婚登记手续。'
# 声明一个Field对象，对象里面填的就是需要对文本进行哪些操作，比如这里lower=True英文大写转小写,tokenize=cut对于文本分词采用之前定义好的cut函数，sequence=True表示输入的是一个sequence类型的数据，还有其他更多操作可以参考文档

f = open("temp_input.tsv",'w')
writer = csv.writer(f,delimiter='\t')
writer.writerow([0,"FUCK",test_string])
f.close()

TEXT = torchtext.legacy.data.Field(sequential=True, lower=True, tokenize=cut) # , fix_length=300
LABEL = torchtext.legacy.data.Field(sequential=False, dtype=torch.int64)
text_field=[('index', None),('label', None), ('content', TEXT)]
label_field=[('index', None),('label', LABEL), ('content', TEXT)]

pretrained_name = 'sgns.renmin.bigram-char'
pretrained_path = '/Users/scarlett/Downloads'
vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=pretrained_path)



text_data = torchtext.legacy.data.TabularDataset(
    'temp_input.tsv',
    format='tsv',
    skip_header=False,
    fields = text_field
)
label_data = torchtext.legacy.data.TabularDataset(
    'cause_label.tsv',
    format='tsv',
    skip_header=False,
    fields = label_field
)
TEXT.build_vocab(text_data, vectors=vectors)
LABEL.build_vocab(label_data)

text_iter, label_iter = torchtext.legacy.data.BucketIterator.splits(
    (text_data,label_data),
    batch_size=32,
    sort_key=lambda x: len(x.content)
)



def predict(text_iter, label_iter):

    textcnn_model.load_state_dict(torch.load('model/bestmodel_steps200.pt'))
    textcnn_model.eval()

    for batch in text_iter:
        feature = batch.content
        logits = textcnn_model(feature)
        print(torch.max(logits, 1)[1])

    for batch in label_iter:
        target = batch.label
        print(target.data)


predict(text_iter,label_iter)