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

    original_string = entity_description


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

def rel_h_t_judge(rel, entity_h_id, entity_t_id, rel_judge_dict, reverse_rel_info, doc_id, df):

    entity_h_type = df['vertexSet'][doc_id][entity_h_id][0]['type']
    entity_t_type = df['vertexSet'][doc_id][entity_t_id][0]['type']
    key = reverse_rel_info[rel]
    if key in rel_judge_dict:
        for limit in rel_judge_dict[key]:
            if limit[0] == entity_h_type and limit[1] == entity_t_type:
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

def get_rel_template_1(rel, reverse_rel_info ,rel2temp):
    if rel != "no_relation":
        key = reverse_rel_info.get(rel)
        rel_template = rel2temp[key]
    else:
        rel_template = rel2temp["no_relation"]


    rel_template_1 = rel_template.replace("<head>", "entity_h_name")

    rel_template_2 = rel_template_1.replace("<tail>", "entity_t_name")

    return rel_template_2

def get_rel_template_2(rel, entity_h, entity_t, reverse_rel_info ,rel2temp):
    if rel != "no_relation":
        key = reverse_rel_info.get(rel)

        rel_template = rel2temp[key]
    else:
        rel_template = rel2temp["no_relation"]


    rel_template_1 = rel_template.replace("<head>", entity_h)
    rel_template_2 = rel_template_1.replace("<tail>", entity_t)

    return rel_template_2

def get_entity_pair(entity_h , entity_t):
    entity_dict = {}
    entity_dict['entity_h_name'] = entity_h
    entity_dict['entity_t_name'] = entity_t
    return entity_dict

def get_example_output(entity_h, entity_t, rel_template):
    output_dict = {}
    output_dict['ans'] = 'YES'
    output_dict['entity_h_name'] = entity_h
    output_dict['entity_t_name'] = entity_t
    output_dict['reason'] = "According to the relation description, " + rel_template
    return output_dict

def get_txt(entity_h, entity_t, entity_h_description, entity_t_description, evidence):
    txt = f"""{entity_h} is {entity_h_description}
{entity_t} is {entity_t_description}
{evidence}"""
    return txt

def get_doc(doc_id, df):

    sentence_str = ""
    for sentence in df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "
    return sentence_str

#------------------------------------------------------------------------------------------
data_name = "dev" # dev  test
#------------------------------------------------------------------------------------------


doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_fr = open(doc_filename, 'r', encoding='utf-8')
json_info = docred_fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

train_doc_filename = f"{doc_dir}train_annotated.json"
train_docred_fr = open(train_doc_filename, 'r', encoding='utf-8')
train_json_info = train_docred_fr.read()
train_docred_df = pd.read_json(train_json_info)


info_fr = open('../data/docred/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}



with open('../relation_description.json', 'r') as file:
    rel_description_dict = json.load(file)

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


entity_information_objects = read_jsonl(f"../data/entity_information/{data_name}/result_docred_{data_name}_entity_information_0-{docred_len}.jsonl")
entity_information_list = {}

for data in entity_information_objects:
    title = data["title"]
    entity = data["entity"]
    response = data["response"]
    if title not in entity_information_list:
        entity_information_list[title] = {}
    if entity not in entity_information_list[title]:
        entity_information_list[title][entity] = ""
    entity_information_list[title][entity] = response


file_path = f"../data/check_result_multiple_choice_jsonl/{data_name}/result_docred_{data_name}_multiple_choice_path-k20_0-{docred_len}.jsonl"
jsonl_data = read_jsonl(file_path)

save_list = []

start = 0
length  = len(jsonl_data)
print(length)

for id in range(start, length):

    data = jsonl_data[id]

    prompt_rel = data["prompt_rel"]
    response = data["response"]
    entity_h = data["entity_h"]
    entity_t = data["entity_t"]
    entity_h_id = data["entity_h_id"]
    entity_t_id = data["entity_t_id"]
    title = data["title"]
    doc_id = get_docid(title, docred_df)
    doc_text = get_doc(doc_id, docred_df)

    cnt = 0
    len_rel = len(prompt_rel)
    rel_list = []
    limit = 0.0
    for rel_id in range(len_rel):
        rel = prompt_rel[rel_id]
        op = chr(65 + cnt)
        if op in response:
            rel_list.append(rel)
        cnt += 1


    if entity_h in entity_information_list[title]:
        entity_h_description = entity_information_list[title][entity_h]
    else:
        entity_h_description = "no description"
    if entity_t in entity_information_list[title]:
        entity_t_description = entity_information_list[title][entity_t]
    else:
        entity_t_description = "no description"

    evidence = get_evidence(doc_id, entity_h, entity_h_id, entity_t, entity_t_id, docred_df)


    entity_h_description = add_period_if_missing(entity_h_description)
    entity_t_description = add_period_if_missing(entity_t_description)
    evidence = add_period_if_missing(evidence)

    entity_h_description = deal_head_description(entity_h, entity_h_description)
    entity_t_description = deal_head_description(entity_t, entity_t_description)

    for rel in rel_list:
        if rel == "no_relation":
            continue

        if rel_h_t_judge(rel, entity_h_id, entity_t_id, rel_judge_dict, reverse_rel_info, doc_id, docred_df):

            extract_txt = get_txt(entity_h, entity_t, entity_h_description, entity_t_description, evidence)
            extract_entity_pair = get_entity_pair(entity_h, entity_t)

            rel_description = rel_description_dict[rel]

            instruction = f"""Based on the text and the description of the relation "{rel}", give an answer about whether the head and tail entity pairs (head entity and tail entity) satisfy the "{rel}" relation."""
            inputs = f"""## Relation description:
{rel_description}

## The text to be extracted:
{doc_text}

## Entity pair to be extracted:
{extract_entity_pair}"""

            prompt1 = get_prompt(instruction, inputs)

            save_dict_1 = {}
            save_dict_1['title'] = title
            save_dict_1['doc_id'] = doc_id
            save_dict_1['prompt'] = prompt1
            save_dict_1["entity_h"] = entity_h
            save_dict_1["entity_t"] = entity_t
            save_dict_1["entity_h_id"] = entity_h_id
            save_dict_1["entity_t_id"] = entity_t_id
            save_dict_1["prompt_rel"] = rel
            save_dict_1["response"] = ""
            save_list.append(save_dict_1)

save_name = f"../data/triplet_fact_judgement_prompt/{data_name}/triplet_fact_judgement_prompt_{data_name}_k20.jsonl"
save_to_jsonl(save_list, save_name)
print(f"The result is saved in the file {save_name}")