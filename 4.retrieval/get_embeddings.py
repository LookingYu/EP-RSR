import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

def deal_head_description(entity , entity_description):

    original_string = entity_description
    prefix = entity + " is "
    if original_string.startswith(prefix):
        result = original_string[len(prefix):]
    else:
        result = original_string

    return result

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def add_period_if_missing(string):

    punctuation = set('!.,?;:')
    string = string.rstrip()
    if string and string[-1] in punctuation:
        return string[:-1] + ','
    else:
        string += ','
    return string

def get_sentence(entity_h, entity_t, entity_h_description, entity_t_description, entities_description):
    sentence = f"""“{entity_h}” is {entity_h_description}“{entity_t}” is {entity_t_description}the relation between “{entity_h}” and “{entity_t}” in the context:{entities_description}"""

    return sentence


#-----------------------------------------------------------------------------------
data_name = "dev"  # train_annotated dev test
#-----------------------------------------------------------------------------------

doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"

fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

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

data_file_path = f"../data/check_result_relation_summary_jsonl/{data_name}/result_docred_{data_name}_relation_summary_0-{docred_len}.jsonl"

data_jsonl_data = read_jsonl(data_file_path)
print(f"{data_file_path} loading successful")

embeddings_file_path = f"../data/get_embeddings/{data_name}_embeddings_0-{docred_len}.npy"

model = SentenceTransformer('../data/all-mpnet-base')
print("model loading successful")


now = datetime.now()

formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("The start time is:", formatted_time)


data_len = len(data_jsonl_data)
print(f"data len: {data_len}")


data_sentences = []
for data_id in range(data_len):
    print(data_id)
    title = data_jsonl_data[data_id]['title']
    entity_h = data_jsonl_data[data_id]['entity_h']
    entity_t = data_jsonl_data[data_id]['entity_t']
    entities_description = data_jsonl_data[data_id]['entities_description']

    if entity_h in entity_information_list[title]:
        entity_h_description = entity_information_list[title][entity_h]
    else:
        entity_h_description = "no description"
    if entity_t in entity_information_list[title]:
        entity_t_description = entity_information_list[title][entity_t]
    else:
        entity_t_description = "no description"

    entity_h_description = add_period_if_missing(entity_h_description)
    entity_t_description = add_period_if_missing(entity_t_description)
    entity_h_description = deal_head_description(entity_h, entity_h_description)
    entity_t_description = deal_head_description(entity_t, entity_t_description)

    sentence = get_sentence(entity_h, entity_t, entity_h_description, entity_t_description, entities_description)

    print(sentence)
    print("--------------------------------------------------")

    data_sentences.append(sentence)

data_embeddings = model.encode(data_sentences)


np.save(embeddings_file_path, data_embeddings)
print(f"{data_name}_embeddings_0-{docred_len} save in {embeddings_file_path}")


now = datetime.now()

formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("The end time is:", formatted_time)