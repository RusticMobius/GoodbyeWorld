import json
import random
import csv

def split_dataset():
    with open('/Users/scarlett/PycharmProjects/GoodbyeWorld/selected_data.json', 'r', encoding='utf-8') as json_file:
        dicts = json.load(json_file)
        all_label = []
        for d in dicts:
            label = d["案由"]
            all_label.append(label)

        all_type = set(all_label)
        all_record = {}
        for type in all_type:
            all_record[type] = []
            count = 0
            for d in dicts:
                if type == d["案由"]:
                    count += 1
                    all_record[type].append(d["案件基本情况"])

    f1 = open("train_data.tsv", 'w')
    writer_train = csv.writer(f1, delimiter='\t')

    f2 = open("test_data.tsv", 'w')
    writer_test = csv.writer(f2, delimiter='\t')

    for key, value in all_record.items():
        random.shuffle(value)
        if len(value) > 500:

            ran_index = random.sample(range(0, len(value)), 500)
            for i in ran_index[0:400]:
                writer_train.writerow([0, key, value[i]])
            for j in ran_index[400:]:
                writer_test.writerow([0, key, value[j]])

        else:

            split_point = round(len(value) * 0.8)
            if split_point >= len(value):
                split_point -= 1

            for i in range(split_point):
                writer_train.writerow([0, key, value[i]])
            for j in range(split_point, len(value)):
                writer_test.writerow([0, key, value[j]])

def label_dataset():
    with open('/Users/scarlett/PycharmProjects/GoodbyeWorld/selected_data.json', 'r', encoding='utf-8') as json_file:
        dicts = json.load(json_file)
        labels = []
        example = []
        f = open("label_data.tsv", 'w')
        writer = csv.writer(f,delimiter='\t')
        for d in dicts:
            if d['案由'] not in labels:
                labels.append(d['案由'])
                example.append([0,d['案由'],'contents'])
        for e in example:
            writer.writerow(e)

import jieba
def seg_dataset_process():
    stopwords1 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
    stopwords2 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
    stopwords3 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
    stopwords4 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
    stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4
    f_train = open('train_data.tsv','r')
    f_test = open('test_data.tsv','r')
    train_set = open('train_seg_data.csv','w')
    test_set = open('test_seg_data.csv','w')

    reader1 = csv.reader(f_train,delimiter='\t')
    reader2 = csv.reader(f_test,delimiter='\t')

    writer1 = csv.writer(train_set,delimiter='\t')
    writer2 = csv.writer(test_set,delimiter='\t')

    for r1 in reader1:
        content = ''
        words = jieba.cut(r1[2])
        for w in words:
            if w not in stopwords:
                content = content + w + ' '
        writer1.writerow([r1[1], content.strip()])
    for r2 in reader2:
        content = ''
        words = jieba.cut(r2[2])
        for w in words:
            if w not in stopwords:
                content = content + w + ' '
        writer2.writerow([r2[1], content.strip()])

seg_dataset_process()