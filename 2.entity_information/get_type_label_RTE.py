import pandas as pd
import numpy as np
import csv
import json

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


file_path1 = f"../data/entity_information/{data_name}/result_{doc_name}_{data_name}_entity_information_0-{docred_len}-type-nodeal.jsonl"
jsonl_data1 = read_jsonl(file_path1)


type_list = ['ORG', 'LOC', 'NUM', 'TIME', 'MISC', 'PER']

save_list = []

jsonl_data = []

for data in jsonl_data1:
    jsonl_data.append(data)

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
    entity = data["entity"]

    input_string = response
    lines = input_string.split('\n')
    results = []
    for line in lines:
        type_name = line.strip()

        if type_name not in type_list:
            continue

        rel_dict = {}
        rel_dict["title"] = title
        rel_dict["entity"] = entity
        rel_dict["type"] = type_name
        save_list.append(rel_dict)

unique_save_list = [dict(t) for t in {tuple(d.items()) for d in save_list}]


save_name = f"../data/entity_information/{data_name}/result_{doc_name}_{data_name}_entity_information_0-{docred_len}-type.jsonl"

save_to_jsonl(unique_save_list, save_name)
print(f"The result is saved in the file {save_name}")
print("-----------------------------------------------")