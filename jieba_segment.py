import csv
import json
import jieba

stopwords1 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
stopwords2 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
stopwords4 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4

f = open('jieba_seg_data.csv','w')
writer = csv.writer(f)
with open('selected_data.json', 'r', encoding='utf-8') as json_file:
    dicts = json.load(json_file)
    for dict in dicts:
        content = [item for item in jieba.lcut(dict['案件基本情况']) if item not in stopwords]
        print(content)
        writer.writerow([dict['案由'],' '.join(content)])