import pandas as pd
import numpy as np
import csv
import json
import re
from sentence_transformers import SentenceTransformer, util

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

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_doc_entitys(doc_id, df):

    entity_list = []
    for entity in df['vertexSet'][doc_id]:
        name_list = []
        for entity_name in entity:
            name = entity_name['name']
            name_list.append(name)

        unique_name_list = list(set(name_list))
        for name in unique_name_list:
            entity_list.append(name)

    return entity_list

def cos(emb1,emb2):
    return util.pytorch_cos_sim(emb1, emb2)

def get_similar_id(entity, entity_list, model, docred_df, doc_id):

    entity_list_embeddings = model.encode(entity_list)

    entity_embeddings = model.encode(entity)

    threshold = 0.9
    ress = cos(entity_list_embeddings, entity_embeddings).reshape(-1).tolist()
    sorted_indices = sorted(range(len(ress)), key=lambda i: ress[i], reverse=True)

    if ress[sorted_indices[0]] < threshold:
        return -1
    else:
        entity_id = get_entity_id(entity_list[sorted_indices[0]], docred_df, doc_id)
        return entity_id


    return -1



#------------------------------------------------------------------------------------------
data_name = "dev" # dev  test
#------------------------------------------------------------------------------------------

doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_fr = open(doc_filename, 'r', encoding='utf-8')
json_info = docred_fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

info_fr = open('../data/docred/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}

save_doc_name = "01-01"

file_path = f"../data/check_result_entity_pair_selection_jsonl/{data_name}/result_docred_{data_name}_entity_pair_selection_0-{docred_len}-{save_doc_name}.jsonl"
jsonl_data = read_jsonl(file_path)


model = SentenceTransformer('../data/all-mpnet-base')

save_list = []

start = 0
length  = len(jsonl_data)
print(length)
cnt = 0

for id in range(start, length):
    print("----------------------------------------")
    print(id)
    data = jsonl_data[id]

    prompt = data["prompt"]
    response = data["response"]
    title = data["title"]
    doc_id = data["doc_id"]


    input_string = response

    lines = input_string.split('\n')

    result = []

    for line in lines:

        parts = line.split('##')

        parts = [part.strip() for part in parts]
        if len(parts) != 2 :
            continue
        first_part = parts[0]
        second_part = parts[1]


        flag = 0
        entity_h = first_part
        entity_t = second_part
        entity_h_id = get_entity_id(entity_h, docred_df, doc_id)
        entity_t_id = get_entity_id(entity_t, docred_df, doc_id)
        if entity_h_id == -1 or entity_t_id == -1:
            print("-----------similar------------")
            entity_list = get_doc_entitys(doc_id, docred_df)
            if entity_h_id == -1:
                entity_h_id = get_similar_id(entity_h, entity_list, model, docred_df, doc_id)
            if entity_t_id == -1:
                entity_t_id = get_similar_id(entity_t, entity_list, model, docred_df, doc_id)
            if entity_h_id == -1 or entity_t_id == -1:
                print("-------------jump------------")
                flag = 1
                cnt += 1

        if flag == 0:
            rel_dict = {}
            rel_dict["title"] = title
            rel_dict['h_idx'] = entity_h_id
            rel_dict['t_idx'] = entity_t_id
            rel_dict["r"] = "P1"
            save_list.append(rel_dict)

            rel_dict = {}
            rel_dict["title"] = title
            rel_dict['h_idx'] = entity_t_id
            rel_dict['t_idx'] = entity_h_id
            rel_dict["r"] = "P1"
            save_list.append(rel_dict)



unique_save_list = [dict(t) for t in {tuple(d.items()) for d in save_list}]


save_name = f"../data/get_entity_pair_selection_label/{data_name}/docred_{data_name}_entity_pair_selection_0-{docred_len}_answer-{save_doc_name}.jsonl"

save_to_jsonl(unique_save_list, save_name)
print(f"The result is saved in the file {save_name}")
print("-----------------------------------------------")
print(cnt)