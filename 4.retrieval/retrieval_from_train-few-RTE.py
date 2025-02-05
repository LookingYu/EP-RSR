import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

def cos(emb1,emb2):
    return util.pytorch_cos_sim(emb1, emb2)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id
def judge_example(reverse_rel_info, tain_docred_df, title, rel, train_entity_h_id, train_entity_t_id):
    train_doc_id = get_docid(title, tain_docred_df)
    label_list = train_docred_df['labels'][train_doc_id]
    rel_id = reverse_rel_info[rel]
    flag = 0
    for label in label_list:
        if label['h'] == train_entity_h_id and label['t'] == train_entity_t_id and label['r'] == rel_id:
            flag = 1
    if flag == 1:
        return True

    return False


data_name = "dev"

doc_name = "redocred"
doc_dir = f'../data/{doc_name}/'
dev_doc_filename = f"{doc_dir}{data_name}_revised.json"
dev_docred_fr = open(dev_doc_filename, 'r', encoding='utf-8')
dev_json_info = dev_docred_fr.read()
dev_docred_df = pd.read_json(dev_json_info)
dev_docred_len = len(dev_docred_df)

train_doc_filename = f"../data/{doc_name}/train_revised.json"
train_docred_fr = open(train_doc_filename, 'r', encoding='utf-8')
train_json_info = train_docred_fr.read()
train_docred_df = pd.read_json(train_json_info)


info_fr = open(f'../data/{doc_name}/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}


train_file_path = f"../data/check_result_relation_summary_jsonl/train/result_{doc_name}_train_relation_summary_0-3053.jsonl"
train_annotated_jsonl_data = read_jsonl(train_file_path)

dev_file_path = f"../data/check_result_relation_summary_jsonl/{data_name}/result_{doc_name}_{data_name}_relation_summary_0-{dev_docred_len}-RTE.jsonl"
dev_jsonl_data = read_jsonl(dev_file_path)

print("data loading successful")
print('--------------------------------------')

train_embeddings = np.load(f'../data/get_embeddings/{doc_name}_train_embeddings_0-3053.npy')
print("train_embeddings loading successful")
print('--------------------------------------')


dev_embeddings = np.load(f'../data/get_embeddings/{doc_name}_{data_name}_embeddings_0-{dev_docred_len}-RTE.npy')
print(f"{data_name}_embeddings loading successful")
print('--------------------------------------')

dev_len = len(dev_jsonl_data)

k = 20

data_sort_n = k
entity_sort_n = k
threshold = 0.0

dev_top_all = {}

print("data len:", dev_len)


for id in range(dev_len):
    print(id)
    query_sentence = dev_jsonl_data[id]['entities_description']
    entity_h_name = dev_jsonl_data[id]["entity_h"]
    entity_t_name = dev_jsonl_data[id]["entity_t"]
    dev_title = dev_jsonl_data[id]['title']

    if dev_title not in dev_top_all:
        dev_top_all[dev_title] = {}

    if entity_h_name not in dev_top_all[dev_title]:
        dev_top_all[dev_title][entity_h_name] = {}
    if entity_t_name not in dev_top_all[dev_title][entity_h_name]:
        dev_top_all[dev_title][entity_h_name][entity_t_name] = []

    if entity_t_name not in dev_top_all[dev_title]:
        dev_top_all[dev_title][entity_t_name] = {}
    if entity_h_name not in dev_top_all[dev_title][entity_t_name]:
        dev_top_all[dev_title][entity_t_name][entity_h_name] = []


    query_embedding = dev_embeddings[id]
    ress = cos(train_embeddings, dev_embeddings[id]).reshape(-1).tolist()

    sorted_indices = sorted(range(len(ress)), key=lambda i: ress[i], reverse=True)
    cnt = 0

    val_len = len(sorted_indices)
    for sort_id in range(val_len):
        train_id = sorted_indices[sort_id]
        if cnt >= data_sort_n:
            break

        if ress[train_id] < threshold:
            break

        for label in train_annotated_jsonl_data[train_id]['label_rel']:
            data_dict = {}
            data_dict['score'] = ress[train_id]
            data_dict['rel'] = label
            data_dict['train_title'] = train_annotated_jsonl_data[train_id]['title']
            if judge_example(reverse_rel_info, train_docred_df, train_annotated_jsonl_data[train_id]['title'], label, train_annotated_jsonl_data[train_id]['entity_h_id'], train_annotated_jsonl_data[train_id]['entity_t_id']):
                data_dict['entity_h'] = train_annotated_jsonl_data[train_id]['entity_h']
                data_dict['entity_t'] = train_annotated_jsonl_data[train_id]['entity_t']
                data_dict['entity_h_id'] = train_annotated_jsonl_data[train_id]['entity_h_id']
                data_dict['entity_t_id'] = train_annotated_jsonl_data[train_id]['entity_t_id']
            else:
                data_dict['entity_h'] = train_annotated_jsonl_data[train_id]['entity_t']
                data_dict['entity_t'] = train_annotated_jsonl_data[train_id]['entity_h']
                data_dict['entity_h_id'] = train_annotated_jsonl_data[train_id]['entity_t_id']
                data_dict['entity_t_id'] = train_annotated_jsonl_data[train_id]['entity_h_id']

            dev_top_all[dev_title][entity_h_name][entity_t_name].append(data_dict)
            dev_top_all[dev_title][entity_t_name][entity_h_name].append(data_dict)


        cnt += 1
    print("-----------------------------")



sorted_dev_top = {}
for key, value in dev_top_all.items():
    sorted_dev_top[key] = {}
    for key_1, value_1 in value.items():
        sorted_dev_top[key][key_1] = {}
        for key_2, value_2 in value_1.items():
            sorted_dev_top[key][key_1][key_2] = []
            sorted_value = sorted(value_2, key=lambda x: x[list(x.keys())[0]], reverse=True)
            sorted_dev_top[key][key_1][key_2] = sorted_value

dev_result_rel = {}

for key, value in sorted_dev_top.items():

    dev_result_rel[key] = {}
    for key_1, value_1 in value.items():
        dev_result_rel[key][key_1] = {}
        for key_2, value_2 in value_1.items():
            dev_result_rel[key][key_1][key_2] = {}
            val_len = len(value_2)
            cnt = 0

            for val_id in range(val_len):

                if value_2[val_id]['rel'] != "no_relation":
                    if value_2[val_id]['rel'] not in dev_result_rel[key][key_1][key_2]:
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']] = {}
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['score'] = value_2[val_id]['score']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['train_title'] = value_2[val_id]['train_title']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_h'] = value_2[val_id]['entity_h']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_t'] = value_2[val_id]['entity_t']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_h_id'] = value_2[val_id]['entity_h_id']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_t_id'] = value_2[val_id]['entity_t_id']

                cnt += 1
                if cnt == entity_sort_n:
                    break


save_result_rel = {}


for key, value in dev_result_rel.items():
    save_result_rel[key] = {}
    for key_1, value_1 in value.items():
        save_result_rel[key][key_1] = {}
        for key_2, value_2 in value_1.items():
            save_result_rel[key][key_1][key_2] = []
            print(f"doc:{key} entity_h:{key_1} entity_t:{key_2} relation number:{len(value_2)}")


            predict_rel_list = {}

            for key_3, value_3 in value_2.items():
                rel = key_3

                if rel not in predict_rel_list:
                    predict_rel_list[rel] = 1
                    save_result_rel[key][key_1][key_2].append((rel, value_3))




save_list = []

for data in dev_jsonl_data:
    entity_h_name = data["entity_h"]
    entity_t_name = data["entity_t"]
    dev_title = data['title']
    data_1 = data
    data_2 = []
    label_list = save_result_rel[dev_title][entity_h_name][entity_t_name]
    for label, value in label_list:
        data_2.append((label, value))
    save_data = []
    save_data.append(data_1)
    save_data.append(data_2)
    save_list.append(save_data)

save_name = f"../data/retrieval_from_train/{data_name}/path-k20-RTE-{doc_name}.jsonl"
save_to_jsonl(save_list, save_name)
print(f"The result is saved in the file {save_name}")
