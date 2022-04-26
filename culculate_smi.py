from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
import logging
import numpy as np
from scipy import linalg
import csv
import pandas as pd

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
#
# sentences = word2vec.LineSentence('sim_data.csv')
# path = get_tmpfile("word2vec.model")
# model = word2vec.Word2Vec(sentences, hs=1, min_count=3, window=5, workers=5)
# model.save("word2vec.model")

model=word2vec.Word2Vec.load('word2vec.model')

def sentence_vector(s):
    print("Im here")
    words = s.split(" ")
    print(words)
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



with open("sim_data.csv", "r", encoding="utf-8") as f:
    contents = f.readlines()
    matrix = np.zeros((len(contents), len(contents)))
    for i in range(len(contents)):
        for j in range(len(contents)):
            # 使用矩阵存储所有案件之间的相似度
            matrix[i][j] = vector_similarity(
                contents[i].strip(), contents[j].strip())

    f1 = open("result.txt", "w", encoding="utf-8")
    for j in range(len(contents)):
        # 获取最为相似的案件
        # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
        index = np.argsort(matrix[j])[-2]

        f1.writelines("案件" + str(j + 1) + ":" + '\t')
        f1.writelines(contents[j])
        f1.writelines("案件" + str(index + 1) + ":" + '\t')
        f1.writelines(contents[index])
        f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')


# def data_pre_process():
#
#     f1 = open("segment_data.csv",'r',encoding='utf-8')
#     f2 = open('sim_data.csv', "w", encoding='utf-8')
#     reader = csv.reader(f1)
#     writer = csv.writer(f2)
#     data = []
#     for line in reader:
#         writer.writerow([line[2]])


