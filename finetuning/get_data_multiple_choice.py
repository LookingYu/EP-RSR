import pandas as pd
import numpy as np
import csv
import json
import pickle

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id


data_name = "train_annotated"

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


rel_objects = read_jsonl(f"../data/multiple_choice_prompt/{data_name}/multiple_choice_prompt-path-k20_{data_name}.jsonl")

save_list = []
num = 0
print(len(rel_objects))

for data in rel_objects:
    num += 1
    print(num)

    instruction = data["instruction"]
    inputs = data["input"]
    output = ""

    choice_rel_list = data["prompt_rel"]
    entity_h = data["entity_h"]
    entity_t = data["entity_t"]
    entity_h_id = data["entity_h_id"]
    entity_t_id = data["entity_t_id"]
    title = data["title"]
    doc_id = get_docid(title, docred_df)
    label_list = docred_df['labels'][doc_id]


    cnt = 0
    flag = 0
    for rel in choice_rel_list:
        op = chr(65 + cnt)
        for label in label_list:
            if rel == "no_relation":
                break
            if label['h'] == entity_h_id and label['t'] == entity_t_id and reverse_rel_info[rel] == label['r']:
                output += op + "\n"
                flag = 1
        cnt += 1
    if flag == 0:
        cnt = 0
        for rel in choice_rel_list:
            op = chr(65 + cnt)
            if rel == "no_relation":
                output += op + "\n"
            cnt += 1


    save_dict = {}
    save_dict["instruction"] = instruction
    save_dict["input"] = inputs
    save_dict["output"] = output
    save_list.append(save_dict)



with open(f'finetuning_data/{data_name}_multiple_choice_data-docred.json', 'w') as json_file:
    json.dump(save_list, json_file, indent=4)

print(f"data len: {len(save_list)}")
print("Data has been saved as a JSON file.")

