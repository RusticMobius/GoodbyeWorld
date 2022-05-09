from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_stopwords():
    stopwords1 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/baidu_stopwords.txt',
                       'r', encoding='utf-8')]
    stopwords2 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/cn_stopwords.txt', 'r',
                       encoding='utf-8')]
    stopwords3 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/hit_stopwords.txt', 'r',
                       encoding='utf-8')]
    stopwords4 = [line.rstrip() for line in
                  open('/Users/scarlett/PycharmProjects/GoodbyeWorld/stopwords/stopwords-master/scu_stopwords.txt', 'r',
                       encoding='utf-8')]
    stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4
    return stopwords
import csv
def train():
    data = []
    label = []
    f1 = open('data2/test_seg_data.csv','r')
    f2 = open('data2/train_seg_data.csv','r')
    reader1 = csv.reader(f1,delimiter='\t')
    reader2 = csv.reader(f2,delimiter='\t' )
    for r in reader1:
        label.append(r[0])
        data.append(r[1])
    for r in reader2:
        label.append(r[0])
        data.append(r[1])

    x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=0, test_size=0.2)

    # bow = CountVectorizer(stop_words=load_stopwords(),max_df=0.35,min_df=0.035,ngram_range=(1,2),max_features=2500) 0.8135593220338984 best
    bow = CountVectorizer(stop_words=load_stopwords(), max_df=0.35, min_df=0.035,ngram_range=(1,2),max_features=2500)
    # bow = TfidfVectorizer(stop_words=load_stopwords(), max_df=0.35, min_df=0.035,ngram_range=(1,2),max_features=2500) 0.7532956685499058
    x_train = bow.fit_transform(x_train)
    x_test = bow.transform(x_test)
    print(bow.vocabulary_)
    clf = MultinomialNB()
    clf.fit(x_train,y_train)
    score = clf.score(x_test,y_test)
    print(score)
train()