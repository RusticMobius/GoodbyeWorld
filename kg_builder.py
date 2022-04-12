import json
from py2neo import Graph
from tqdm import tqdm
import threading



class GraphBuilder (object):
    def __init__(self):
        super(GraphBuilder, self).__init__()
        self.graph = Graph(
            host=  "127.0.0.1",
            http_port = 7474,
            user = "neo4j",
            password = "123456"
        )

        self.writ_infos = []
        #节点
        self.number = [] #文书编号
        self.id = [] #id
        self.type = []
        #self.plaintiff_team = [] #
        self.plaintiff = []
        self.plaintiff_lawyer = []
        #self.defendant_team = [] #
        self.defendant = []
        self.defendant_lawyer = []
        self.time = []
        self.court = []
        self.chief_justice = []
        self.court_clerk = []
        self.jury = []
        self.rules = []
        self.case_name = []
        self.court_judgement = []
        self.court_decision = []

        # 关系
        self.rels_number = []
        self.rels_type = []
        self.rels_court = []
        self.rels_time = []
        self.rels_case_name = []
        self.rels_chief_justice = []
        self.rels_court_clerk = []
        self.rels_jury = []
        self.rels_rules = []
        self.rels_defendant = []
        self.rels_defendant_lawyer = []
        self.rels_plaintiff = []
        self.rels_plaintiff_lawyer = []
        self.rels_court_decision = []
        self.rels_court_judgement = []

    def event_extractor(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as json_file:
            dicts = json.load(json_file)
            for dict in dicts:
                id = str(dict["id"])
                self.id.append(id)

                if(len(dict["number"])!=0):
                    self.number.append(dict["number"])
                    self.rels_number.append([id,'文书编号',dict["number"]])

                if (len(dict["case_name"]) != 0):
                    self.case_name.append(dict["case_name"])
                    self.rels_case_name.append([id,"案件名称",dict["case_name"]])

                if (len(dict["time"]) != 0):
                    self.time.append(dict["time"])
                    self.rels_time.append([id,"文书时间",dict["time"]])

                if (len(dict["type"]) != 0):
                    self.type.append(dict["type"])
                    self.rels_type.append([id,'文书类型',dict["type"]])

                if (len(dict["court"]) != 0):
                    self.court.append(dict["court"])
                    self.rels_court.append([id,'法院信息',dict["court"]])

                if (len(dict["chief_justice"]) != 0):
                    self.chief_justice.append(dict["chief_justice"])
                    self.rels_chief_justice.append([id,'审判长',dict["chief_justice"]])

                if (len(dict["court_clerk"]) != 0):
                    self.court_clerk.append(dict["court_clerk"])
                    self.rels_court_clerk.append([id,"书记员",dict["court_clerk"]])

                if (len(dict["court_decision"]) != 0):
                    self.court_decision.append(dict["court_decision"])
                    self.rels_court_decision.append([id,'判决结果',dict["court_decision"]])

                if (len(dict["court_judgement"]) != 0 ):
                    self.court_judgement.append(dict["court_judgement"])
                    self.rels_court_judgement.append([id,'法院认定',dict["court_judgement"]])

                for j in dict["jury"]:
                    self.jury.append(j)
                    self.rels_jury.append([id,"人民陪审员",j])

                for d in dict["defendant"]:
                    self.defendant.append(d)
                    self.rels_defendant.append([id, '被告', d])

                for dl in dict["defendant_lawyer"]:
                    self.defendant_lawyer.append(dl)
                    self.rels_defendant_lawyer.append([id, '被告委托代理人', dl])

                for p in dict["plaintiff"]:
                    self.plaintiff.append(p)
                    self.rels_plaintiff.append([id, '原告', p])

                for pl in dict["plaintiff_lawyer"]:
                    self.plaintiff_lawyer.append(pl)
                    self.rels_plaintiff_lawyer.append([id,'原告委托代理人',pl])

                for r in dict["rules"]:
                    self.rules.append(r)
                    self.rels_rules.append([id,'参考法条',r])

                self.writ_infos.append(dict)


    def write_nodes(self,entitys,entity_type):
        for node in tqdm(entitys,ncols=80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type, entity_name=node.replace("'", ""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                # print(cql)

    def write_edges(self, triples, head_type, tail_type):
        for head, relation, tail in tqdm(triples, ncols=80):
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                head_type=head_type, tail_type=tail_type, head=head.replace("'", ""),
                tail=tail.replace("'", ""), relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
               # print(cql)


    def set_attributes(self, entity_infos, etype):
        print("写入 {0} 实体的属性".format(etype))
        for e_dict in tqdm(entity_infos, ncols=80):

            id = str(e_dict['id'])
            for k, v in e_dict.items():

                cql = """MATCH (n:{label})
                                        WHERE n.name='{name}'
                                        set n.{k}={v}""".format(label=etype, name=id.replace("'",""), k=k, v=v)

                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    # print(cql)

    def create_entitys(self):
        self.write_nodes(self.id,'文书ID')
        self.write_nodes(self.number,'文书编号')
        self.write_nodes(self.case_name,'案名')
        self.write_nodes(self.type,'文书类型')
        self.write_nodes(self.court,'法院')
        self.write_nodes(self.time,'文书时间')
        self.write_nodes(self.chief_justice,'审判长')
        self.write_nodes(self.court_clerk,'书记员')
        self.write_nodes(self.defendant,'被告')
        self.write_nodes(self.jury,'人民陪审员')
        self.write_nodes(self.defendant_lawyer,'被告委托代理人')
        self.write_nodes(self.plaintiff,'原告')
        self.write_nodes(self.plaintiff_lawyer,'原告委托代理人')
        self.write_nodes(self.rules,'参考法条')
        self.write_nodes(self.court_decision,'法院认定')
        self.write_nodes(self.court_judgement,'判决结果')


    def create_relations(self):
        self.write_edges(self.rels_defendant, '文书ID', '被告')
        self.write_edges(self.rels_defendant_lawyer, '文书ID', '被告委托代理人')
        self.write_edges(self.rels_plaintiff, '文书ID', '原告')
        self.write_edges(self.rels_plaintiff_lawyer, '文书ID', '原告委托代理人')
        self.write_edges(self.rels_number,'文书ID','文书编号')
        self.write_edges(self.rels_type,'文书ID','文书类型')
        self.write_edges(self.rels_time,'文书ID','文书时间')
        self.write_edges(self.rels_case_name,'文书ID','案名')
        self.write_edges(self.rels_court,'文书ID','法院')
        self.write_edges(self.rels_court_decision, '文书ID', '判决结果')
        self.write_edges(self.rels_court_judgement, '文书ID', '法院认定')
        self.write_edges(self.rels_chief_justice, '文书ID', '审判长')
        self.write_edges(self.rels_court_clerk, '文书ID', '书记员')
        self.write_edges(self.rels_jury, '文书ID', '人民陪审员')
        self.write_edges(self.rels_rules, '文书ID', '参考法条')



    def set_writ_attributes(self):
        # self.set_attributes(self.disease_infos,"疾病")
        t = threading.Thread(target=self.set_attributes, args=(self.writ_infos, "文书ID"))
        t.setDaemon(False)
        t.start()


if __name__ == '__main__':

    path = 'data.json'
    builder = GraphBuilder()
    builder.event_extractor(path)
    builder.create_entitys()
    builder.create_relations()
    builder.set_writ_attributes()
