import pandas as pd
import numpy as np
import csv
import json

def get_rel_template(rel, entity_h, entity_t, reverse_rel_info ,rel2temp):
    if rel != "no_relation":
        key = reverse_rel_info.get(rel)
        rel_template = rel2temp[key]
    else:
        rel_template = rel2temp["no_relation"]

    rel_template_1 = rel_template.replace("<head>", entity_h)
    rel_template_2 = rel_template_1.replace("<tail>", entity_t)

    return rel_template_2

def get_doc_entitys(doc_id, df):

    entity_list = []
    for entity in df['vertexSet'][doc_id]:

        name = entity[0]['name']

        entity_list.append(name)

    return entity_list

def get_entity_id(entity, df, doc_id):

    len_doc = len(df['vertexSet'][doc_id])

    for entity_id in range(len_doc):

        for entity_name in df['vertexSet'][doc_id][entity_id]:
            if entity_name['name'] == entity:
                return entity_id

    return -1


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

    instruction = f"""Given a text and an entity list as input, list the entity pairs that can be identified as possibly containing a relation."""
    input = f"""## Text:
{sentence_str}

## Entity list:
{entity_list}"""
    output = ""


    label_list = docred_df['labels'][doc_id]


    for label in label_list:
        name_h = docred_df['vertexSet'][doc_id][label['h']][0]['name']
        name_t = docred_df['vertexSet'][doc_id][label['t']][0]['name']
        str_entities = f"{name_h} ## {name_t}\n"
        output += str_entities

    save_dict = {}
    save_dict["instruction"] = instruction
    save_dict["input"] = input
    save_dict["output"] = output
    save_list.append(save_dict)


with open(f'finetuning_data/{data_name}_entity_pair_selection_data-docred.json', 'w') as json_file:
    json.dump(save_list, json_file, indent=4)

print(f"data len: {len(save_list)}")
print("Data has been saved as a JSON file.")


