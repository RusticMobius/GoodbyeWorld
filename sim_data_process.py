# -*- coding: UTF-8 -*-
from ltp import LTP
import csv
from py2neo import Graph
import pandas as pd
import json

ltp = LTP()

stopwords1 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8')]
stopwords2 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/cn_stopwords.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/hit_stopwords.txt', 'r', encoding='utf-8')]
stopwords4 = [line.rstrip() for line in
                  open('stopwords/stopwords-master/scu_stopwords.txt', 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4

def data_select_from_database():
    graph = Graph(
        host="127.0.0.1",
        http_port=7474,
        user="neo4j",
        password="123456"
    )
    data = []
    result = graph.run("MATCH p=(n:`案号`)-[r:`案件基本情况`]->(m:`案件基本情况`) RETURN n,m ").data()
    count = 1
    print(len(result))
    for r in result:
        print(count)
        count += 1
        id =r['n']['name']
        segment = stop_words_filter(r['m']['name'])
        data.append([id,segment])

    df = pd.DataFrame(data, columns=['案号','案件基本情况分词结果'])

    df.to_csv("segment_data.csv",index=True,encoding="utf-8-sig")

def select_data_from_json():
    data = []
    count = 1
    with open('selected_data.json', 'r', encoding='utf-8') as json_file:
        dicts = json.load(json_file)
        for d in dicts:
            print(count)
            count += 1
            id = str(d["案号"])
            segment = stop_words_filter(str(d['案件基本情况']))
            print(segment)
            data.append([id,segment])
    df = pd.DataFrame(data, columns=['案号','案件基本情况分词结果'])
    df.to_csv("segment_data.csv",index=True,encoding="utf-8-sig")

def stop_words_filter(ori_str):
    data = [ori_str]
    segment = ltp.seg(data)
    result = ""
    for item in segment[0][0]:
        if item not in stopwords:
            result = result + item + " "
    return result

if __name__ == '__main__':
    select_data_from_json()