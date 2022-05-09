# -*- coding: UTF-8 -*-
import flask
import json
import cal_mini
import sim_data_process
from textcnnClassify import predict, textcnn_model
import torch
from flask import Flask, jsonify, request
from py2neo import Graph
from flask_cors import CORS

app = Flask(__name__)
graph = Graph('http://localhost:7474', auth=('neo4j', '123456'))
model = textcnn_model
model.load_state_dict(torch.load('model1/bestmodel_steps320(b32).pt'))
predict(model, "SOS")

# CORS(app, supports_credentials=True)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type,token'


@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Credentials'] = 'true'
    environ.headers['Access-Control-Allow-Origin'] = request.environ['HTTP_ORIGIN']
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With,token'
    return environ


# @app.route('/calculateSimiliarity', methods=['post'])
# def calculate_similarity():
#     recv_data = request.args.get("info")
#
#
#     if recv_data:
#         print(recv_data)
#         case_info = sim_data_process.stop_words_filter(recv_data)
#         return cal_mini.predict(case_info)
#     else:
#         print("receive None")
#         return "ERROR"


@app.route('/matchCaseInfo', methods=['post'])
def match_case_info():
    recv_data = request.args.get("info")

    if recv_data:
        print(recv_data)
        # json_re = json.loads(recv_data)
        case_info = sim_data_process.stop_words_filter(recv_data)
        id_list = cal_mini.predict(case_info)
        case_info_list = {}
        for id in id_list.keys():
            quest = """MATCH (n:`案号`)-[r:`案件基本情况`]->(m:`案件基本情况`)
                                    WHERE n.name='{id}'
                                    RETURN m""".format(id=id)
            result = graph.run(quest).data()
            for r in result:
                case_info_list[id] = r['m']['name'] + "(相似度：" + id_list[id] + ")"
        return case_info_list

    else:
        print("receive None")
        return "ERROR in getting caseInfo"


@app.route('/matchCaseJudgement', methods=['post'])
def match_case_judgement():
    recv_data = request.args.get("info")

    if recv_data:
        print(recv_data)
        # json_re = json.loads(recv_data)
        case_info = sim_data_process.stop_words_filter(recv_data)
        id_list = cal_mini.predict(case_info)
        case_type_list = {}
        for id in id_list.keys():
            case_type_list[id] = {}
            quest1 = """MATCH (n:`案号`)-[r:`裁判结果`]->(m:`裁判结果`)
                                                            WHERE n.name='{id}'
                                                            RETURN m""".format(id=id)
            quest2 = """MATCH (n:`案号`)-[r:`案件基本情况`]->(m:`案件基本情况`)
                                                            WHERE n.name='{id}'
                                                            RETURN m""".format(id=id)
            result1 = graph.run(quest1).data()
            result2 = graph.run(quest2).data()
            case_type_list[id]["相似度"] = id_list[id]
            for r in result2:
                case_type_list[id]["案件基本情况"] = r['m']['name']
            for r in result1:
                case_type_list[id]["裁判结果"] = r['m']['name']

        return case_type_list

    else:
        print("receive None")
        return "ERROR in getting caseType"


@app.route('/matchLegalItem', methods=['post'])
def match_legal_item():
    recv_data = request.args.get("info")

    if recv_data:
        print(recv_data)
        # json_re = json.loads(recv_data)
        case_info = sim_data_process.stop_words_filter(recv_data)
        id_list = cal_mini.predict(case_info)
        legal_item_list = {}
        for id in id_list.keys():
            quest = """MATCH (n:`案号`)-[r:`法条`]->(m:`法条`)
                                        WHERE n.name='{id}'
                                        RETURN m""".format(id=id)
            result = graph.run(quest).data()
            for r in result:
                legal_item_list[id] = r['m']['name'] + "(相似度：" + id_list[id] + ")"
        return legal_item_list

    else:
        print("receive None")
        return "ERROR in getting legalItem"


@app.route('/matchCauseType', methods=['post'])
def match_cause_type():
    recv_data = request.args.get("info")
    if recv_data:
        # print(recv_data)
        cause_type = predict(model, recv_data)
        print(cause_type)
    return cause_type


app.run(host='localhost', port=8802, debug=True)
