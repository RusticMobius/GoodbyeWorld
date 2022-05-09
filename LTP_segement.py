from ltp import LTP
import csv
import json

ltp = LTP()

f1 = open ('ltp_seg_data.csv', 'w', encoding="utf-8")
writer1 = csv.writer(f1, delimiter='\t')


stopwords1 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
stopwords2 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
stopwords4 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4

with open('selected_data.json', 'r', encoding='utf-8') as json_file:
    dicts = json.load(json_file)
    for dict in dicts:
        segment = ltp.seg([dict['案件基本情况']])
        meaningful_words = []
        for item in segment[0][0]:
            if item not in stopwords:
                meaningful_words.append(item)

        writer1.writerow([dict['案由'],' '.join(meaningful_words)])