import pandas as pd
import numpy as np
import csv
import json
import re
import ast
from fuzzywuzzy import fuzz
import itertools
from sentence_transformers import SentenceTransformer, util

def get_doc_title(doc_id, df):

    title = df['title'][doc_id]
    return title

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

def sliding_window_fuzzy_match(entity, text):
    """
    :param entity:
    :param text:
    :return:
    """
    target_list = entity.split()
    max_similarity = 0
    best_match = ""
    target_combinations = list(itertools.combinations(target_list, len(target_list)))
    words = text.split()
    for window_size in range(len(target_list), len(words) + 1):
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            for target_combination in target_combinations:
                target_string = " ".join(target_combination)
                similarity_ratio = fuzz.ratio(window, target_string)
                if similarity_ratio > max_similarity:
                    if window.startswith(",") or window.startswith("."):
                        continue
                    if window.endswith(",") or window.endswith("("):
                        window = window[:-1]
                    max_similarity = similarity_ratio
                    best_match = window
    if max_similarity > 60:
        return best_match.strip()
    else:
        return ""

def get_fixed_entity(entity, sentence):
    """
        将entity进行修正
    :param ori_entities:
    :param sentence:
    :return:
    """

    if entity and entity.strip() in sentence:
        return entity.strip()
    else:
        fixed_entity = sliding_window_fuzzy_match(entity.strip(), sentence).strip()
        if fixed_entity:
            if fixed_entity.strip() in sentence:
                return fixed_entity.strip()
        else:
            return ""


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

save_doc_name = "01-RTE"

file_path = f"../data/check_result_entity_pair_selection_jsonl/{data_name}/result_{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}-{save_doc_name}.jsonl"
jsonl_data = read_jsonl(file_path)

print(save_doc_name)
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

    sentence_str = ""
    for sentence in docred_df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "

    input_string = response

    try:
        entity_json = ast.literal_eval(input_string)
    except (SyntaxError, ValueError) as e:
        entity_json = {}
        print(f"转换失败: {e}")
    for key, value in entity_json.items():
        flag_h = -1

        entity_h_name_fixed = get_fixed_entity(key, sentence_str)
        if entity_h_name_fixed != "":
            flag_h = 1

        if flag_h == -1:
            print(f"key:{key} not real")
            continue
        for entity_value in value:
            flag_t = -1

            entity_t_name_fixed = get_fixed_entity(entity_value, sentence_str)
            if entity_t_name_fixed != "":
                flag_t = 1

            if flag_t == 1:
                entity_h_name = entity_h_name_fixed
                entity_t_name = entity_t_name_fixed

                if entity_h_name == entity_t_name:
                    continue

                rel_dict = {}
                rel_dict["title"] = title
                rel_dict['h_name'] = entity_h_name
                rel_dict['t_name'] = entity_t_name
                rel_dict["r"] = "P1"
                save_list.append(rel_dict)

                rel_dict = {}
                rel_dict["title"] = title
                rel_dict['h_name'] = entity_t_name
                rel_dict['t_name'] = entity_h_name
                rel_dict["r"] = "P1"
                save_list.append(rel_dict)


unique_save_list = [dict(t) for t in {tuple(d.items()) for d in save_list}]


save_name = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}_answer-{save_doc_name}.jsonl"

save_to_jsonl(unique_save_list, save_name)
print(f"The result is saved in the file {save_name}")
print("-----------------------------------------------")
print(cnt)