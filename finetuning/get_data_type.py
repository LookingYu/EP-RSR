import pandas as pd
import numpy as np
import csv
import json

def get_doc_entitys(doc_id, df):

    entity_list = []
    for entity in df['vertexSet'][doc_id]:
        # name = entity[0]['name']
        for entity_ok in entity:
            name = entity_ok['name']
            entity_list.append(name)

    return entity_list

def get_entity_type(entity, df, doc_id):

    types = []
    len_doc = len(df['vertexSet'][doc_id])

    for entity_id in range(len_doc):
        for entity_name in df['vertexSet'][doc_id][entity_id]:
            if entity_name['name'] == entity:
                if entity_name['type'] not in types:
                    types.append(entity_name['type'])

    return types

data_name = "train" #train_annotated

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

type_list = []

start = 0
end = docred_len

save_list = []
entity_dict_list = []

for doc_id in range(start, end):
    sentence_str = ""
    for sentence in docred_df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "

    entity_list = get_doc_entitys(doc_id, docred_df)

    entity_type = {}
    for entity in entity_list:
        types = get_entity_type(entity, docred_df, doc_id)

        if entity not in entity_type:
            entity_type[entity] = []

        for type in types:
            if type not in type_list:
                type_list.append(type)

            if type not in entity_type[entity]:
                entity_type[entity].append(type)

    for entity in entity_list:
        instruction = f"""Given a text as input, extract related type information of entity "{entity}" from the text."""
        input = f"""## Text:
{sentence_str}"""
        output = str("\n".join([str(type) for type in entity_type[entity]]))


        save_dict = {}
        save_dict["instruction"] = instruction
        save_dict["input"] = input
        save_dict["output"] = output
        save_list.append(save_dict)


with open(f'finetuning_data/{data_name}_type_data-{doc_name}.json', 'w') as json_file:
    json.dump(save_list, json_file, indent=4)

print(f"data len: {len(save_list)}")
print("Data has been saved as a JSON file.")


