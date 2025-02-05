import json
import pandas as pd
import subprocess
import numpy as np
import json
import re
import csv

def get_promot_txt(filename):

    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')


def get_prompt(instruction, inputs):
    prompt = f"""{instruction}
{inputs}"""
    return prompt

def deal_head_description(entity , entity_description):
    original_0 = entity_description
    prefix_0 = "The answer is "
    if original_0.startswith(prefix_0):
        entity_description_2 = original_0[len(prefix_0):]
    else:
        entity_description_2 = original_0


    original_string = entity_description_2

    prefix = entity + " is "

    if original_string.startswith(prefix):
        result = original_string[len(prefix):]
    else:
        result = original_string

    return result


def add_period_if_missing(string):

    string = string.rstrip()

    if string and string[-1] != '.':
        string += '.'
    return string

def rel_h_t_judge(rel, type_dict, entity_h, entity_t, rel_judge_dict, reverse_rel_info, doc_id, df):

    entity_h_type = type_dict[entity_h]
    entity_t_type = type_dict[entity_t]

    key = reverse_rel_info[rel]
    if key in rel_judge_dict:
        for limit in rel_judge_dict[key]:
            if limit[0] in entity_h_type and limit[1] in entity_t_type:
                return True

        return False
    else:
        return False

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id

def get_evidence(doc_id, entity_h, entity_h_id, entity_t, entity_t_id, df):
    sentence_str_all = ""
    sent_id_list = []

    flag = 0
    for entity_h_dict in df['vertexSet'][doc_id][entity_h_id]:
        for entity_t_dict in df['vertexSet'][doc_id][entity_t_id]:
            if entity_h_dict['name'] == entity_h:
                if entity_t_dict['name'] == entity_t:
                    if entity_h_dict['sent_id'] == entity_t_dict['sent_id']:
                        flag = 1
                        sent_id_list.append(entity_h_dict['sent_id'])

    if flag == 0:

        for entity_h_dict in df['vertexSet'][doc_id][entity_h_id]:
            if entity_h_dict['name'] == entity_h:
                sent_id_list.append(entity_h_dict['sent_id'])

        for entity_t_dict in df['vertexSet'][doc_id][entity_t_id]:
            if entity_t_dict['name'] == entity_t:
                sent_id_list.append(entity_t_dict['sent_id'])

    unique_sorted_list = sorted(set(sent_id_list))
    list_len = len(unique_sorted_list)
    for i in range(list_len):
        sent_id =  unique_sorted_list[i]
        sentence_str = ""
        for word in df['sents'][doc_id][sent_id]:
            sentence_str += word
            sentence_str += " "
        sentence_str_all += sentence_str

    return sentence_str_all


def read_list_text_file_to_list(file_path):

    text_data = []

    with open(file_path, 'r') as file:
        for line in file:
            text_data.append(eval(line))

    return text_data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_csv(csv_file):
    data_list = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_list.append(row)
    return data_list

def get_rel_template(rel, entity_h, entity_t, reverse_rel_info ,rel2temp):
    if rel != "no_relation":
        key = reverse_rel_info.get(rel)
        rel_template = rel2temp[key]
    else:
        rel_template = rel2temp["no_relation"]


    rel_template_1 = rel_template.replace("<head>", entity_h)
    rel_template_2 = rel_template_1.replace("<tail>", entity_t)

    return rel_template_2

def get_doc(doc_id, df):

    sentence_str = ""
    for sentence in df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "
    return sentence_str



data_name = "dev"
doc_name = "redocred"

rel_objects = read_jsonl(f"../data/retrieval_from_train/{data_name}/path-k20-RTE-{doc_name}.jsonl")

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

with open('../rel2temp_with_1.json', 'r') as file:
    rel2temp = json.load(file)


rel_judge_file_path = "../rel_judge.txt"
rel_judge_list = read_list_text_file_to_list(rel_judge_file_path)
rel_judge_list = rel_judge_list[0]
rel_judge_dict = {}
for list in rel_judge_list:

    fruits = list[0].split('_')
    if fruits[0] not in rel_judge_dict:
        rel_judge_dict[fruits[0]] = [(fruits[1], fruits[2])]
    else:
        rel_judge_dict[fruits[0]].append((fruits[1], fruits[2]))


file_path = f"../data/entity_information/{data_name}/result_{doc_name}_{data_name}_entity_information_0-{docred_len}-type.jsonl"
jsonl_data = read_jsonl(file_path)

type_dict = {}
for data in jsonl_data:
    entity = data["entity"]
    type = data["type"]
    title = data["title"]

    if title not in type_dict:
        type_dict[title] = {}
    if entity not in type_dict[title]:
        type_dict[title][entity] = []

    if type not in type_dict[title][entity]:
        type_dict[title][entity].append(type)




print(len(rel_objects))
print("-----------------------------------------")
start = 0

length = len(rel_objects)


save_list = []

for id in range(start, length):

    dev_dict = {}
    rels_list = []
    cnt = 0
    for item in rel_objects[id]:
        if cnt == 0:
            dev_dict = item
        if cnt == 1:
            for rel_item in item:
                rels_list.append(rel_item)
        cnt += 1

    entities_description = dev_dict["entities_description"]
    entity_h = dev_dict["entity_h"]
    entity_t = dev_dict["entity_t"]
    title = dev_dict["title"]
    doc_id = get_docid(title, docred_df)
    doc_text = get_doc(doc_id, docred_df)


    choice_rel_list = []
    for rel_dict in rels_list:

        rel = rel_dict[0]
        choice_rel_list.append(rel)


    choice_rel_list_h_t = []
    for rel in choice_rel_list:
        # choice_rel_list_h_t.append(rel)
        if entity_h not in type_dict[title] or entity_t not in type_dict[title]:
            choice_rel_list_h_t.append(rel)
        else:
            if rel_h_t_judge(rel, type_dict[title], entity_h, entity_t, rel_judge_dict, reverse_rel_info, doc_id, docred_df):
                choice_rel_list_h_t.append(rel)

    if len(choice_rel_list_h_t) > 0 :
        choice_rel_list_h_t.append("no_relation")

        options = ""
        cnt = 0
        for rel in choice_rel_list_h_t:
            op = chr(65 + cnt)
            op_temp = get_rel_template(rel, entity_h, entity_t, reverse_rel_info, rel2temp)
            options += op + ". " + op_temp + '\n'
            cnt += 1

        instruction = f"""Determine which option can be inferred from the given text."""
        inputs = f"""## Text:
{doc_text}

## Options:
{options}"""

        prompt1 = get_prompt(instruction, inputs)

        save_dict_1 = {}
        save_dict_1["instruction"] = instruction
        save_dict_1["input"] = inputs
        save_dict_1["output"] = ""
        save_dict_1['title'] = title
        save_dict_1['doc_id'] = doc_id
        save_dict_1['prompt'] = prompt1
        save_dict_1["entity_h"] = entity_h
        save_dict_1["entity_t"] = entity_t
        save_dict_1["prompt_rel"] = choice_rel_list_h_t
        save_dict_1["response"] = []
        save_list.append(save_dict_1)


    choice_rel_list_t_h = []
    for rel in choice_rel_list:
        # choice_rel_list_t_h.append(rel)
        if entity_h not in type_dict[title] or entity_t not in type_dict[title]:
            choice_rel_list_t_h.append(rel)
        else:
            if rel_h_t_judge(rel, type_dict[title], entity_t, entity_h, rel_judge_dict, reverse_rel_info, doc_id, docred_df):
                choice_rel_list_t_h.append(rel)

    if len(choice_rel_list_t_h) > 0 :
        choice_rel_list_t_h.append("no_relation")

        options = ""
        cnt = 0
        for rel in choice_rel_list_t_h:
            op = chr(65 + cnt)
            op_temp = get_rel_template(rel, entity_t, entity_h, reverse_rel_info, rel2temp)
            options += op + ". " + op_temp + '\n'
            cnt += 1


        instruction = f"""Determine which option can be inferred from the given text."""
        inputs = f"""## Text:
{doc_text}

## Options:
{options}"""

        prompt2 = get_prompt(instruction, inputs)

        save_dict_2 = {}
        save_dict_2["instruction"] = instruction
        save_dict_2["input"] = inputs
        save_dict_2["output"] = ""
        save_dict_2['title'] = title
        save_dict_2['doc_id'] = doc_id
        save_dict_2['prompt'] = prompt2
        save_dict_2["entity_h"] = entity_t
        save_dict_2["entity_t"] = entity_h
        save_dict_2["prompt_rel"] = choice_rel_list_t_h
        save_dict_2["response"] = []
        save_list.append(save_dict_2)


save_name = f"../data/multiple_choice_prompt/{data_name}/multiple_choice_prompt-path-k20_{data_name}-RTE-{doc_name}.jsonl"
save_to_jsonl(save_list, save_name)
print(f"The result is saved in the file {save_name}")
print(len(save_list))

