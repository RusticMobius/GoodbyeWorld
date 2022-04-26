import pandas as pd
import json

path = 'selected_data.json'
# json_file = open(path, 'r', encoding='utf-8')
df = pd.read_json(path)
# dicts = json.load(json_file)

data = df[["案由", "案件基本情况"]]
data.columns = ['label', 'content']
print(data.head())

lhjf = data.loc[data['label'] == '离婚纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
syjf = data.loc[data['label'] == '赡养纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fyfjf = data.loc[data['label'] == '抚养费纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
jcjf = data.loc[data['label'] == '继承纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
lhhccjf = data.loc[data['label'] == '离婚后财产纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fdjcjf = data.loc[data['label'] == '法定继承纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
syfjf = data.loc[data['label'] == '赡养费纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
bgfygxjf = data.loc[data['label'] == '变更抚养关系纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fyjf = data.loc[data['label'] == '抚养纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
hyccjf = data.loc[data['label'] == '婚约财产纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fyfjf_ = data.loc[data['label'] == '扶养费纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
yzjcjf = data.loc[data['label'] == '遗嘱继承纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
twqjf = data.loc[data['label'] == '探望权纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fjxcjf = data.loc[data['label'] == '分家析产纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
jhqjf = data.loc[data['label'] == '监护权纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
jcsygxjf = data.loc[data['label'] == '解除收养关系纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
tjgxjf = data.loc[data['label'] == '同居关系纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
tjgxxcjf = data.loc[data['label'] == '同居关系析产纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fyjf_ = data.loc[data['label'] == '扶养纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
hywxjf = data.loc[data['label'] == '婚姻无效纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
hyjtjf = data.loc[data['label'] == '婚姻家庭纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
fqccydjf = data.loc[data['label'] == '夫妻财产约定纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
dwjcjf = data.loc[data['label'] == '代位继承纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
bgsygxjf = data.loc[data['label'] == '变更赡养关系纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
zjcjf = data.loc[data['label'] == '转继承纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
cxhyjf = data.loc[data['label'] == '撤销婚姻纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)
sygxjf = data.loc[data['label'] == '收养关系纠纷'].sample(frac=.8, replace=False, random_state=0, axis=0)

frames = [lhjf, syjf, fyfjf, jcjf, lhhccjf, fdjcjf, syfjf, bgfygxjf, fyjf, hyccjf, fyfjf_, yzjcjf, twqjf, fjxcjf, jhqjf,
          jcsygxjf, tjgxjf, tjgxxcjf, fyjf_, hywxjf, hyjtjf, fqccydjf, dwjcjf, bgsygxjf
    ,zjcjf, cxhyjf, sygxjf
          ]

train_data = pd.concat(frames)

res_data = data[~data.index.isin(train_data.index)]

_lhjf = res_data.loc[res_data['label'] == '离婚纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_syjf = res_data.loc[res_data['label'] == '赡养纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fyfjf = res_data.loc[res_data['label'] == '抚养费纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_jcjf = res_data.loc[res_data['label'] == '继承纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_lhhccjf = res_data.loc[res_data['label'] == '离婚后财产纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fdjcjf = res_data.loc[res_data['label'] == '法定继承纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_syfjf = res_data.loc[res_data['label'] == '赡养费纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_bgfygxjf = res_data.loc[res_data['label'] == '变更抚养关系纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fyjf = res_data.loc[res_data['label'] == '抚养纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_hyccjf = res_data.loc[res_data['label'] == '婚约财产纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fyfjf_ = res_data.loc[res_data['label'] == '扶养费纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_yzjcjf = res_data.loc[res_data['label'] == '遗嘱继承纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_twqjf = res_data.loc[res_data['label'] == '探望权纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fjxcjf = res_data.loc[res_data['label'] == '分家析产纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_jhqjf = res_data.loc[res_data['label'] == '监护权纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_jcsygxjf = res_data.loc[res_data['label'] == '解除收养关系纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_tjgxjf = res_data.loc[res_data['label'] == '同居关系纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_tjgxxcjf = res_data.loc[res_data['label'] == '同居关系析产纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fyjf_ = res_data.loc[res_data['label'] == '扶养纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_hywxjf = res_data.loc[res_data['label'] == '婚姻无效纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_hyjtjf = res_data.loc[res_data['label'] == '婚姻家庭纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_fqccydjf = res_data.loc[res_data['label'] == '夫妻财产约定纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_dwjcjf = res_data.loc[res_data['label'] == '代位继承纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_bgsygxjf = res_data.loc[res_data['label'] == '变更赡养关系纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_zjcjf = res_data.loc[res_data['label'] == '转继承纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_cxhyjf = res_data.loc[res_data['label'] == '撤销婚姻纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)
_sygxjf = res_data.loc[res_data['label'] == '收养关系纠纷'].sample(frac=.5, replace=False, random_state=0, axis=0)

frames = [
_lhjf, _syjf, _fyfjf, _jcjf, _lhhccjf, _fdjcjf, _syfjf, _bgfygxjf, _fyjf, _hyccjf, _fyfjf, _yzjcjf, _twqjf, _fjxcjf,
_jhqjf, _jcsygxjf, _tjgxjf, _tjgxxcjf, _fyjf_, _hywxjf, _hyjtjf, _fqccydjf, _dwjcjf, _bgsygxjf
    , _zjcjf, _cxhyjf,_sygxjf
]
test_data = pd.concat(frames)
dev_data = res_data[~res_data.index.isin(test_data.index)]

train_data.to_csv('./data/train_data.csv',sep="\t")
test_data.to_csv('./data/test_data.csv',sep="\t")
dev_data.to_csv('./data/dev_data.csv',sep="\t")

print(train_data)

from collections import Counter

print('总数据：')
print(Counter(list(data.label)))
print('=================================')
print('训练集：')
print(Counter(list(train_data.label)))
print('=================================')
print('测试集：')
print(Counter(list(test_data.label)))