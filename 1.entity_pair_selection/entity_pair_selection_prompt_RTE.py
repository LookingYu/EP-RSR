import pandas as pd
import numpy as np
import csv
import json

def get_prompt(instruction, inputs):
    prompt = f"""{instruction}
{inputs}"""
    return prompt

def get_doc_title(doc_id, df):

    title = df['title'][doc_id]
    return title

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_doc(doc_id, df):

    sentence_str = ""
    for sentence in df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "
    return sentence_str


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

start = 0
end = docred_len

save_list = []

for doc_id in range(start, end):

    title = get_doc_title(doc_id, docred_df)
    sentence_str = get_doc(doc_id, docred_df)

    instruction = f"""Given a text as input, list the entity pairs that can be identified as possibly containing a relation."""
    inputs = f"""## Text:
{sentence_str}"""

    prompt = get_prompt(instruction, inputs)

    save_dict = {}
    save_dict['title'] = title
    save_dict['doc_id'] = doc_id
    save_dict["instruction"] = instruction
    save_dict["input"] = inputs
    save_dict['prompt'] = prompt
    save_dict["response"] = ""
    save_list.append(save_dict)


save_name = f"../data/entity_pair_selection_prompt/{data_name}/entity_pair_selection_prompt_{data_name}_01_RTE_{doc_name}.jsonl"

save_to_jsonl(save_list, save_name)
print(f"The result is saved in the file {save_name}")