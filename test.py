import json

with open('data.json', 'r', encoding="utf-8") as load_f:
    dicts = json.load(load_f)
    for d in dicts:
        print(d)