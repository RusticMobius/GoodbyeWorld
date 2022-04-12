# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import re
import os
import json
import csv

def re_extract(file, id):
    f = open(path+"/"+file)
    str_number = "(?<=)（\d{4}）.*?号"
    str_time = "[\u4e00-\u9fa5〇]+?年[\u4e00-\u9fa5〇]+?月[\u4e00-\u9fa5〇]+?日"
    str_court_name = "^(.*?)人民法院"
    str_judger = "(审\s*判\s*长)(.*)"
    str_writer = '(书\s*记\s*员)(.*)'
    str_person = '(人民陪审员)(.*)'
    str_plaintiff = "(?<=\s原告)([\u4e00-\u9fa5]+)(?=，)"
    str_plaintiff_lawyer = "原告[\s\S]+?委托代理人(.*)，.*?律师"
    str_defendant = "(?<=\s被告)([\u4e00-\u9fa5]+)(?=，)"
    str_defendant_lawyer = "被告[\s\S]+?委托代理人(.*)，.*?律师"
    str_case_name = "原告.*?被告.*?一案"
    str_law_accordings = "依照(.*)?(?=，判决)"
    str_file_type = "民 事.*?书"
    iter_f = iter(f)
    context = ""
    judgement = ""
    decision = ""
    addFlag = False
    for line in  iter_f:
        if(line.startswith("本院认为")):
            # print(line)
            judgement = judgement + line
            addFlag = True
            continue
        if(addFlag):
            if(line!="\n" and line!="（此页无正文）\n"):
                decision = decision + line
            else:
                addFlag = False

        context = context + line
    #print(decision)

    if(judgement=="" and decision==""):
        return None

    number = re.findall(str_number, context)
    time = re.findall(str_time, context)
    court_name = re.findall(str_court_name, context)
    judger = re.findall(str_judger, context)
    writer = re.findall(str_writer, context)
    plaintiff = re.findall(str_plaintiff, context)
    plaintiff_lawyer = re.findall(str_plaintiff_lawyer, context)
    defendant = re.findall(str_defendant, context)
    defendant_lawyer = re.findall(str_defendant_lawyer, context)
    case_name = re.findall(str_case_name, context)
    laws = re.findall(str_law_accordings, context)
    type = re.findall(str_file_type, context)
    person = re.findall(str_person, context)

    data = {}
    data["id"] = id
    data["number"] = ""
    data["time"] = ""
    data["court"] = ""
    data["case_name"] = ""
    data["chief_justice"] = ""
    data["court_clerk"] = []
    data["type"] = ""
    data["jury"] = []
    data["plaintiff"] = []
    data["plaintiff_lawyer"] = []
    data["defendant"] = []
    data["defendant_lawyer"] = []
    data["rules"] = []
    data["court_judgement"] = ""
    data["court_decision"] = ""

    data["court_judgement"] = judgement
    data["court_decision"] = decision



    if (len(number) != 0):
        # print("文书编号：" + number[0])
        data["number"] = number[0].strip()
    else:
        return None

    if (len(time) != 0):
        # print("文书时间" + time[0])
        data["time"] = time[0].strip()

    if (len(court_name)!=0):
        # print("法院：" + court_name[0])
        data["court"] = court_name[0].strip() + "人民法院"

    if (len(judger)!=0):
        # print("审判长:")
        # print(judger[0][1])
        data["chief_justice"] = judger[0][1].replace(" ","")

    if (len(writer)!=0):
        # print("书记员：")
        # print(writer[0][1])
        data["court_clerk"] = writer[0][1].replace(" ","")

    if (len(plaintiff)!=0):
        # print("原告:")
        # print(plaintiff)
        for p in plaintiff:
            if (p.strip("：") not in data["plaintiff"]):
                if ("诉称" not in p.strip()):
                    data["plaintiff"].append(p.strip("："))
    else:
        return None

    if (len(plaintiff_lawyer)!=0):
        # print("原告委托代理人" )
        # print(plaintiff_lawyer)
        for pl in plaintiff_lawyer:
            if (pl.strip("：") not in data["plaintiff_lawyer"]):
                data["plaintiff_lawyer"].append(pl.strip("："))

    if (len(defendant) != 0):
        # print("被告")
        # print( defendant)
        for d in defendant:
            if (d.strip("：") not in data["defendant"]):
                if("辩称" not in d.strip()):
                    data["defendant"].append(d.strip("："))

    if (len(defendant_lawyer) != 0):
        # print("被告委托代理人" )
        # print(defendant_lawyer)
        for dl in defendant_lawyer:
            if (dl.strip("：") not in data["defendant_lawyer"]):
                data["defendant_lawyer"].append(dl.strip("："))
    else:
        return None

    if (len(case_name) != 0):
        # print("案件" + case_name[0])
        data["case_name"] = case_name[0]
    else:
        return None

    if (len(laws) != 0):
        # print("参考法条")
        # print( laws)
        for r in laws:
            if (r not in data["rules"]):
                data["rules"].append(r)

    if (len(type) != 0):
        # print("文书类型" + type[0])
        data["type"] = type[0].replace(" ","")

    if (len(person) != 0):
        for j in person:
            if (j[1].strip() not in data["jury"]):
                data["jury"].append(j[1].strip())



    print(data)
    return data


if __name__ == '__main__':
    # path = '/Users/scarlett/PycharmProjects/dataPrePro/testData'
    path =  '/Users/scarlett/Downloads/raw'
    files = os.listdir(path)
    count = 0
    data_list = []
    context_list = []
    with open('data.json', 'w', encoding="utf-8") as f:  # writing Json data
        for file in files:
            context = {}
            count = count + 1
            data = re_extract(file, count)
            if(data!=None):
                data_list.append(data)
                context["id"] = data["id"]
                context["text"] = data["court_judgement"].replace("\n","")
                context_list.append(context)

        d = json.dumps(data_list,indent=4,ensure_ascii=False)
        f.write(d)
        f.close()
    with open('processed_data', 'w', encoding="utf-8") as f1:
        writer = csv.writer(f1, delimiter=' ')
        for context in context_list:
            writer.writerow([context["id"],context["text"]])
