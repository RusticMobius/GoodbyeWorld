from xml.dom.minidom import parse
import xml.dom.minidom
import os
import json
import csv

def readXML(path):
    print(path)
    domTree = parse(path)
    collection = domTree.documentElement
    # 文首信息
    data = {}
    # data["经办法院"] = ""
    # data["文书名称"] = ""
    # data["案号"] = ""
    # data["案件性质"] = ""
    # data["案件类型"] = ""
    # data["案由"] = ""
    # data["案件基本情况"] = ""
    # data["裁判分析过程"] = ""
    data["法条"] = []
    # data["裁判结果"] = ""
    print(collection.getElementsByTagName("WS")[0].getElementsByTagName("JBFY")[0].getAttribute("value"))
    try:
        ws = collection.getElementsByTagName("WS")[0]

        try:
            jbfy = ws.getElementsByTagName("JBFY")[0].getAttribute("value")
            data["经办法院"] = jbfy + ""
        except Exception as e:
            pass

        try:
            wsmc = ws.getElementsByTagName("WSMC")[0].getAttribute("value")
            data["文书名称"] = wsmc + ""
        except:
            pass
        try:
            ah = ws.getElementsByTagName("AH")[0].getAttribute("value")
            data["案号"] = ah + ""
        except:
            pass
        try:
            ajxz = ws.getElementsByTagName("AJXZ")[0].getAttribute("value")
            data["案件性质"] = ajxz + ""
        except:
            pass
        try:
            ajlx = ws.getElementsByTagName("AJLX")[0].getAttribute("value")
            data["案件类型"] = ajlx + ""
        except:
            pass
    except:
        return
    # print(jbfy,wsmc,ah,ajxz,ajlx)

    # 案由
    try:
        ssjl = collection.getElementsByTagName("SSJL")[0]
        try:
            ay = ssjl.getElementsByTagName("AY")[0].getAttribute("value")
        except:
            data["案由"] = ay + ""
            pass
    except:
        return

    # 案件基本情况
    try:
        ajjbqk = collection.getElementsByTagName("AJJBQK")[0].getAttribute("value")
        data["案件基本情况"] = ajjbqk + ""
    except:
        return

    # 裁判分析过程
    try:
        cpfxgc = collection.getElementsByTagName("CPFXGC")[0].getAttribute("value")
        data["裁判分析过程"] = cpfxgc + ""
    except:
        return

    #法条

    try:
        flftmc_list = collection.getElementsByTagName("CPFXGC")[0].getElementsByTagName("FLFTMC")
        flftmc = []
        for flft in flftmc_list:
            tm_list = flft.getElementsByTagName("TM")
            for item in tm_list:
                k_list = item.getElementsByTagName("KM")
                if len(k_list) == 0:
                    flftmc.append(flft.getAttribute("value") + "第" + item.getAttribute("value") + "条")
                else:
                    for k in k_list:
                        flftmc.append(flft.getAttribute("value") + "第" + item.getAttribute("value") + "条" + "第" + k.getAttribute("value"))
        for ft in flftmc:
            data["法条"].append(ft + "")
    except:
        return
    # 裁判结果

    try:
        cpjg = collection.getElementsByTagName("CPJG")[0].getAttribute("value")
        data["裁判结果"] = cpjg + ""
    except:
        return

    return data
def initial():
    path = '/Users/scarlett/Downloads/毕设数据集'
    folders = os.listdir(path)
    files = []
    folders_path = []
    for folder in folders:
        if "." not in folder:
            folders_path.append(path + "/" + folder)
    for folder_path in folders_path:
        for file_name in os.listdir(folder_path):
            files.append(folder_path + "/" + file_name)
    count = 0
    data_list = []
    print(len(files))
    with open('selected_data.json', 'w', encoding="utf-8") as f:  # writing Json data
        for file in files:

            try:
                data = readXML(file)
                # print(data)
                if (data):
                    data_list.append(data)
            except:
                count += 1
                print(path)

        d = json.dumps(data_list, indent=4, ensure_ascii=False)
        f.write(d)
        f.close()

        print(len(data_list),count)


initial()