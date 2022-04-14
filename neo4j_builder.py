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
        self.jbfy = [] #经办法院
        self.wsmc = [] #文书名称
        self.ah = [] #案号
        self.ajxz = [] #案件性质
        self.ajlx = [] #案件类型
        self.ay = [] #案由
        self.ajjbqk = [] #案件基本情况
        self.cpfxgc = [] #裁判分析过程
        self.ft = [] #法条
        self.cpjg = [] #裁判结果

        # 关系
        self.rels_jbfy = []
        self.rels_wsmc = []
        self.rels_ajxz = []
        self.rels_ajlx = []
        self.rels_ay = []
        self.rels_ajjbqk = []
        self.rels_cpfxgc = []
        self.rels_ft = []
        self.rels_cpjg = []

    def event_extractor(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as json_file:
            dicts = json.load(json_file)
            for dict in dicts:
                id = str(dict["案号"])
                self.ah.append(id)

                if(len(dict["经办法院"])!=0):
                    self.jbfy.append(dict["经办法院"])
                    self.rels_jbfy.append([id,'经办法院',dict["经办法院"]])

                if (len(dict["文书名称"]) != 0):
                    self.wsmc.append(dict["文书名称"])
                    self.rels_wsmc.append([id,"文书名称",dict["文书名称"]])

                if (len(dict["案件性质"]) != 0):
                    self.ajxz.append(dict["案件性质"])
                    self.rels_ajxz.append([id,"案件性质",dict["案件性质"]])

                if (len(dict["案件类型"]) != 0):
                    self.ajlx.append(dict["案件类型"])
                    self.rels_ajlx.append([id,'案件类型',dict["案件类型"]])

                if (len(dict["案件基本情况"]) != 0):
                    self.ajjbqk.append(dict["案件基本情况"])
                    self.rels_ajjbqk.append([id,'案件基本情况',dict["案件基本情况"]])

                if (len(dict["裁判分析过程"]) != 0):
                    self.cpfxgc.append(dict["裁判分析过程"])
                    self.rels_cpfxgc.append([id,'裁判分析过程',dict["裁判分析过程"]])

                if (len(dict["案由"]) != 0):
                    self.ay.append(dict["案由"])
                    self.rels_ay.append([id,'案由',dict["案由"]])

                if (len(dict["裁判结果"]) != 0):
                    self.cpjg.append(dict["裁判结果"])
                    self.rels_cpjg.append([id,"裁判结果",dict["裁判结果"]])

                for j in dict["法条"]:
                    self.ft.append(j)
                    self.rels_ft.append([id,"法条",j])

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
        self.write_nodes(self.jbfy,'经办法院')
        self.write_nodes(self.wsmc,'文书名称')
        self.write_nodes(self.ah,'案号')
        self.write_nodes(self.ajxz,'案件性质')
        self.write_nodes(self.ajlx,'案件类型')
        self.write_nodes((self.ay,'案由'))
        self.write_nodes(self.ajjbqk,'案件基本情况')
        self.write_nodes(self.cpfxgc,'裁判分析过程')
        self.write_nodes(self.ft,'法条')
        self.write_nodes(self.cpjg,'裁判结果')

    def create_relations(self):
        self.write_edges(self.rels_jbfy, '案号', '经办法院')
        self.write_edges(self.rels_wsmc, '案号', '文书名称')
        self.write_edges(self.rels_ajxz, '案号', '案件性质')
        self.write_edges(self.rels_ajlx, '案号', '案件类型')
        self.write_edges(self.rels_ay,'案号','案由')
        self.write_edges(self.rels_ajjbqk,'案号','案件基本情况')
        self.write_edges(self.rels_cpfxgc,'案号','裁判分析过程')
        self.write_edges(self.rels_ft,'案号','法条')
        self.write_edges(self.rels_cpjg,'案号','裁判结果')



    def set_writ_attributes(self):
        # self.set_attributes(self.disease_infos,"疾病")
        t = threading.Thread(target=self.set_attributes, args=(self.writ_infos, "案号"))
        t.setDaemon(False)
        t.start()


if __name__ == '__main__':

    path = 'selected_data.json'
    builder = GraphBuilder()
    builder.event_extractor(path)
    builder.create_entitys()
    builder.create_relations()
    # builder.set_writ_attributes()
