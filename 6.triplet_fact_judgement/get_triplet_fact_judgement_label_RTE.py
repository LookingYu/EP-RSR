import pandas as pd
import numpy as np
import csv
import json

def get_doc_title(doc_id, df):

    title = df['title'][doc_id]

    return title


def get_entity_id(entity, df, doc_id):
    len_doc = len(df['vertexSet'][doc_id])
    for entity_id in range(len_doc):
        for entity_name in df['vertexSet'][doc_id][entity_id]:
            if entity_name['name'] == entity:
                return entity_id

    return -1

def judge_rel(entity_h_id, entity_t_id, rel , doc_id, docred_df):

    for label in docred_df['labels'][doc_id]:
        if entity_h_id == label['h'] and entity_t_id == label['t'] and rel == label['r']:
            return True

    return False

def read_csv(csv_file):
    data_list = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_list.append(row)
    return data_list

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii = False)
            jsonlfile.write('\n')
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data



data_name = "dev"
doc_name = "redocred"

doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}_revised.json"
docred_fr = open(doc_filename, 'r', encoding='utf-8')
json_info = docred_fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

info_fr = open(f'../data/{doc_name}/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}

save_doc_name = f"k20-RTE-{doc_name}"

file_path = f"../data/check_result_triplet_fact_judgement_jsonl/{data_name}/result_{doc_name}_{data_name}_triplet_fact_judgement_0-{docred_len}-{save_doc_name}.jsonl"

jsonl_data = read_jsonl(file_path)


save_list = []

start = 0
length  = len(jsonl_data)

for id in range(start, length):

    data = jsonl_data[id]

    prompt_rel = data["prompt_rel"]
    prompt = data["prompt"]
    response = data["response"]
    entity_h_name = data["entity_h"]
    entity_t_name = data["entity_t"]
    doc_id = data["doc_id"]


    if "YES" in response:

        if data["entity_h"] == data["entity_t"]:
            continue

        rel_dict = {}
        rel_dict["title"] = get_doc_title(doc_id, docred_df)
        rel_dict['h_name'] = data["entity_h"]
        rel_dict['t_name'] = data["entity_t"]
        rel_dict["r"] = reverse_rel_info[prompt_rel]
        save_list.append(rel_dict)

unique_save_list = [dict(t) for t in {tuple(d.items()) for d in save_list}]

save_name = f"../data/get_triplet_fact_judgement_label/{data_name}/{doc_name}_{data_name}_triplet_fact_judgement_0-{docred_len}_answer-{save_doc_name}.jsonl"
save_to_jsonl(unique_save_list, save_name)
print(f"The result is saved in the file {save_name}")
print("-----------------------------------------------")
