from ltp import LTP
import csv

ltp = LTP()

f1 = open ('data_segment', 'w', encoding="utf-8")
f2 = open ('processed_data', 'r', encoding="utf-8")
writer1 = csv.writer(f1, delimiter=' ')
reader2 = csv.reader(f2, delimiter=' ')

stopwords1 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
stopwords2 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
stopwords4 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4

for data in reader2:
    list = [data[1]]
    segment = ltp.seg(list)
    meaningful_words = [data[0]]
    for item in segment[0][0]:
        if item not in stopwords:
            meaningful_words.append(item)
    print(meaningful_words)
    writer1.writerow(meaningful_words)