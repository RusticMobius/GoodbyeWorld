from gensim.models import word2vec
import logging
import numpy as np
from scipy import linalg
import csv

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

sentences = word2vec.LineSentence('segment_train')
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=5)

def sentence_vector(s):
    words = s.split(" ")
    v = np.zeros(100)
    for word in words:
        try:
            v += model.wv[word]
        except:
            continue
    v /= len(words)
    return v

def vector_similarity(s1, s2):
    '''
    计算两个句子之间的相似度:将两个向量的夹角余弦值作为其相似度
    :param s1:
    :param s2:
    :return:
    '''
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))

f1 = open("segment_train", 'r', encoding="utf-8")
f2 = open("segment_test", 'r', encoding="utf-8")
f3 = open("result.txt", 'w', encoding="utf-8")

reader1 = csv.reader(f1, delimiter=' ')
reader2 = csv.reader(f2, delimiter=' ')

contents1_index = []
contents2_index = []
contents1 = []
contents2 = []
str1 = ""
str2 = ""

for item in reader1:
    for i in range(len(item)):
        if i==0:
            contents1_index.append(item[i])
        else:
            str1 = str1 + item[i] + ' '
    contents1.append(str1)

for item in reader2:
    for i in range(len(item)):
        if i == 0:
            contents2_index.append(item[i])
        else:
            str2 = str2 + item[i] + ' '
    contents2.append(str2)

matrix = np.zeros((len(contents2), len(contents1)))

for i in range(len(contents1)):
    for j in range(len(contents2)):
        matrix[j][i] = vector_similarity(
            contents2[j].strip(), contents1[i].strip()
        )



for k in range(len(contents2)):
    index = np.argsort(matrix[k])[-1]
    f3.writelines("案件" + contents2_index[k] + ":" + '\t')
    f3.writelines(contents2[k])
    f3.writelines("案件" + contents1_index[index] + ":" + '\t')
    f3.writelines(contents1[index])
    f3.writelines("相似度：" + str(matrix[k][index]) + '\n\n')





