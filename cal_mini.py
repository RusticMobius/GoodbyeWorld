# -*- coding: UTF-8 -*-
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
import logging
import numpy as np
from scipy import linalg
import sim_data_process
from py2neo import Graph

graph = Graph('http://localhost:7474', auth = ('neo4j', '123456'))

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)




model=word2vec.Word2Vec.load('word2vec.model1')

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




def predict(case_info):
    #type 1 case info
    #type 2 case_type
    #type 3 legal_item
    with open("segment_data.csv", "r", encoding="utf-8") as f:
        file = f.readlines()
        contents = []
        id = []

        for i in range(len(file)):
            if i == 0:
                continue
            f_list = file[i].split(",")
            contents.append(f_list[2])
            id.append(f_list[1])

        matrix = np.zeros((1, len(contents)))

        for j in range(len(contents)):
            # 使用矩阵存储所有案件之间的相似度

            matrix[0][j] = vector_similarity(
                case_info.strip(), contents[j].strip())

        f1 = open("result.txt", "w", encoding="utf-8")

        # 获取最为相似的案件
        # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
        result = {}
        rank = -2
        for i in range(3):
            index = np.argsort(matrix[0])[rank]
            rank -= 1
            result[id[index]] = str(matrix[0][index])

        return result

def match_case_info():
    id_list = "（2006）东民初字第1312号\n（2004）东民初字第357号\n（2008）东民初字第2389号".split('\n')
    case_info_list = {}
    for id in id_list:
        quest = """MATCH (n:`案号`)-[r:`案件基本情况`]->(m:`案件基本情况`)
                            WHERE n.name='{id}'
                            RETURN m""".format(id=id)
        result = graph.run(quest).data()
        for r in result:
            case_info_list[id] = r['m']['name']
    print(case_info_list)

def match_case_type():
    id_list = "（2006）东民初字第1312号\n（2004）东民初字第357号\n（2008）东民初字第2389号".split('\n')
    case_type_list = {}
    for id in id_list:
        quest = """MATCH (n:`案号`)-[r:`案由`]->(m:`案由`)
                                WHERE n.name='{id}'
                                RETURN m""".format(id=id)
        result = graph.run(quest).data()
        for r in result:
            case_type_list[id] = r['m']['name']
    print(case_type_list)

def match_legal_item():
    id_list = "（2006）东民初字第1312号\n（2004）东民初字第357号\n（2008）东民初字第2389号".split('\n')
    legal_item_list = {}
    for id in id_list:
        quest = """MATCH (n:`案号`)-[r:`法条`]->(m:`法条`)
                                    WHERE n.name='{id}'
                                    RETURN m""".format(id=id)
        result = graph.run(quest).data()
        for r in result:
            legal_item_list[id] = r['m']['name']
    print(legal_item_list)

def model_test():
    ori_info = "原告诉称，原、被告于2006年经人介绍相识并确立恋爱关系，同年12月底举行婚礼同居生活。2010年2月2日补办结婚登记手续。在同居期间即2007年由双方共同收入投资建造了四间正房和三间厢房，当年对正房进行了装修并搬进居住，2009年对厢房进行了装修。2012年5月原、被告离婚。对同居期间的共同财产并未分割。由于所建造的房屋没有房产证，不宜进行分割，故起诉要求：一、请求人民法院依法确认原、被告双方所建造的宝坻区宝平街道石佛营村东部9排23号四间正房和三间厢房归双方共同共有和共同使用；二、诉讼费由双方均担。 被告辩称，宝平街道石佛营村东部9排23号住房系于克良、于克忠哥俩祖传遗留房产，所拆砖瓦木材全部用在建筑新房所用，于克良在砖厂上班期间所购红砖也用于建筑新房，所以此房系于克忠、于克良共同共有，原告于2006年正月筹备建房，所有材料全部备齐，建房当初设计大面积建房，因与东邻协商不成才在2007年再建。厢房是盖正房和老房的材料共同建造；2009年10月因被告女儿需要住房，对厢房进行了简装共花费3000元，故不同意原告诉讼请求。 第三人述称，宝平街道石佛营村东部9排23号住房系于克良、于克忠共同所有的房产，与原告无关，故不同意原告的诉讼请求。 经审理查明，原、被告于2006年4-5月间经人介绍相识并确立恋爱关系，同年12月28日举行婚礼同居生活。2010年2月2日补办结婚登记手续。双方因感情不和于2012年5月经本院调解离婚。2007年农历3月建造石佛营村东部9排23号正房4间、西厢房3间。该房屋未取得产权证。原告主张该房屋系双方同居期间共同建造，系双方共同财产；被告主张系被告与其兄于克良共同建造，系被告与于克良的共同财产。双方于2009年10月对厢房进行了装修，原告主张支付装修款2000元，被告主张支付20000余元。 对双方的争议问题，原告向本院提供以下证据： 1、帐号02-24010046124546号中国农业银行银行卡支取明 细表1份，证明2007年5月至2008年4月支取款情况，但未提供该帐号系何人所有，亦未能证明取款用途。被告否认该卡取款与盖房有因果关系。 2、天津市宝坻区益利康居装饰材料经营部2011年9月13 出具的收款收据1份，证明支付装饰材料款和工款17500元，但未注明客户名称。被告对以上证据予以否认。 3、梁仕虎、梁仕辉、韩俊松于2012年5月23日出具的书面证明，证明2009年秋后装（修）石佛营村厢房三间支付装修款2000元。 被告向本院提供以下证据： 1、证人郭月出庭证明其作建筑工作，2006年1月其帮助被告购买了钢筋、水泥、木头、白灰、沙子等建筑材料，盖房所需要的东西基本都买齐了。原告对以上证言予以否认。 2、证人郭树旺出庭证明，2006年春，被告盖房买砖，是其为被告拉的砖和瓦。原告对以上证言予以否认。"
    info = sim_data_process.stop_words_filter(ori_info)
    print(predict(info))

# def data_pre_process():
#
#     f1 = open("segment_data.csv",'r',encoding='utf-8')
#     f2 = open('sim_data.csv', "w", encoding='utf-8')
#     reader = csv.reader(f1)
#     writer = csv.writer(f2)
#     data = []
#     for line in reader:
#         writer.writerow([line[2]])


