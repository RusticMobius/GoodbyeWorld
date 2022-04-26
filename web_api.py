# -*- coding: UTF-8 -*-
import flask
import json
import cal_mini
import sim_data_process
from flask import Flask, jsonify, request
from py2neo import Graph

app = Flask(__name__)
graph = Graph('http://localhost:7474', auth = ('neo4j', '123456'))

@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin']='*'
    environ.headers['Access-Control-Allow-Method']='*'
    environ.headers['Access-Control-Allow-Headers']='x-requested-with,content-type'
    return environ

@app.route('/calculateSimiliarity', methods = ['post'])
def calculate_similarity():
    recv_data = request.args.get("info")

    if recv_data:
        print(recv_data)
        case_info = sim_data_process.stop_words_filter(recv_data)
        return cal_mini.predict(case_info)
    else:
        print("receive None")
        return "ERROR"


@app.route('/matchCaseInfo', methods = ['post'])
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


@app.route('/matchCaseType', methods = ['post'])
def match_case_type():
    recv_data = request.args.get("info")

    if recv_data:
        print(recv_data)
        # json_re = json.loads(recv_data)
        case_info = sim_data_process.stop_words_filter(recv_data)
        id_list = cal_mini.predict(case_info)
        case_type_list = {}
        for id in id_list.keys():
            quest = """MATCH (n:`案号`)-[r:`案由`]->(m:`案由`)
                                                            WHERE n.name='{id}'
                                                            RETURN m""".format(id=id)
            result = graph.run(quest).data()
            for r in result:
                case_type_list[id] = r['m']['name'] + "(相似度：" + id_list[id] + ")"

        return case_type_list

    else:
        print("receive None")
        return "ERROR in getting caseType"


@app.route('/matchLegalItem', methods = ['post'])
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


app.run(host='localhost', port=8802, debug=True)