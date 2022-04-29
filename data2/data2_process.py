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

label_dataset()