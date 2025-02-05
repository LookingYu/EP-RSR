import pandas as pd
import numpy as np
import csv
import json

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii = False)
            jsonlfile.write('\n')

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id


data_name = "dev"

doc_name = "redocred"
doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}_revised.json"
fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

info_fr = open(f'../data/{doc_name}/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}

save_doc_name = f"k20-RTE-{doc_name}"
jsonl_filename = f"../data/get_triplet_fact_judgement_label/{data_name}/{doc_name}_{data_name}_triplet_fact_judgement_0-{docred_len}_answer-{save_doc_name}.jsonl"

result_jsonl= [json.loads(i) for i in open(jsonl_filename, encoding='utf-8').readlines()]
result_dict = {}

for data in result_jsonl:
    title = data['title']
    if title not in result_dict:
        result_dict[title] = []
    result_dict[title].append(data)


label_dict = {}
for doc_id in range(docred_len):
    title = docred_df['title'][doc_id]
    sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(docred_df['sents'][doc_id])]])
    label_list = docred_df['labels'][doc_id]
    same_fact_list = []

    for label in label_list:
        same_fact = []
        relation = rel_info[label['r']]
        for h in docred_df['vertexSet'][doc_id][label['h']]:
            if h['name'] not in sentence:
                h['name'] = " ".join(docred_df['sents'][doc_id][h['sent_id']][h['pos'][0]:h['pos'][1]])
        for t in docred_df['vertexSet'][doc_id][label['t']]:
            if t['name'] not in sentence:
                t['name'] = " ".join(docred_df['sents'][doc_id][t['sent_id']][t['pos'][0]:t['pos'][1]])
        for h in docred_df['vertexSet'][doc_id][label['h']]:
            for t in docred_df['vertexSet'][doc_id][label['t']]:
                if (h['name'] , relation, t['name']) not in same_fact:
                    same_fact.append([h['name'], relation, t['name']])

        same_fact_list.append(same_fact)

    label_dict[title] = {"same_fact_list": same_fact_list}


my_result = {}
for key, value in result_dict.items():
    title = key
    dev_docid = get_docid(title, docred_df)
    fact_index = []
    wrong = []
    right = []
    ori_fact_list = []

    for label in value:
        fact = [label["h_name"], rel_info[label["r"]], label["t_name"]]
        if fact[0] == fact[2]:
            continue
        ori_fact_list.append(fact)

    for fact in ori_fact_list:
        flag = 0
        for index, true_fact in enumerate(label_dict[title]['same_fact_list']):
            if fact in true_fact:
                flag = 1
                if index not in fact_index:
                    right.append(fact)
                    fact_index.append(index)
        if not flag:
            wrong.append(fact)

    miss = [s_f_l for i, s_f_l in enumerate(label_dict[title]['same_fact_list']) if i not in fact_index]

    my_result[title] = {}
    my_result[title]["right_fact_list"] = right
    my_result[title]["wrong_fact_list"] = wrong
    my_result[title]["miss_fact_list"] = miss
    my_result[title]["true_fact_list"] = label_dict[title]['same_fact_list']


all_up = 0
all_down_label = 0
all_down_predict = 0

for key, value in my_result.items():
    title = key
    data = value

    if "right_fact_list" in data:
        for fact in data["right_fact_list"]:
            all_up += 1
            all_down_predict += 1
            all_down_label += 1
    if "wrong_fact_list" in data:
        for fact in data["wrong_fact_list"]:
            all_down_predict += 1
    if "miss_fact_list" in data:
        for fact in data["miss_fact_list"]:
            all_down_label += 1

all_pre = all_up / all_down_predict
print(f"pre:{all_pre}")
all_rec = all_up / all_down_label
print(f"rec:{all_rec}")
all_F1 = 2*all_pre*all_rec / (all_pre+all_rec)
print(f"F1:{all_F1}")