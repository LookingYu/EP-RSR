from get_docred_doc import get_doc
from get_docred_doc_title import get_doc_title
from get_prompt import get_prompt_entity_rel
import pandas as pd
import json

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii=False)
            jsonlfile.write('\n')

def get_doc_entitys(doc_id, df):

    entity_list = []

    for entity in df['vertexSet'][doc_id]:
        name_list = []
        for entity_name in entity:
            name = entity_name['name']
            name_list.append(name)

        unique_name_list = list(set(name_list))

        entity_list.append(unique_name_list)

    return entity_list

def get_entity_id(entity, df, doc_id):

    len_doc = len(df['vertexSet'][doc_id])

    for entity_id in range(len_doc):
        for entity_name in df['vertexSet'][doc_id][entity_id]:
            if entity_name['name'] == entity:
                return entity_id

    return -1

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:

            data.append(json.loads(line))
    return data


data_name = "dev"

doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"
fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

ok_entitys_dict = {}

if data_name == "train_annotated" or data_name == "train":
    for doc_id in range(docred_len):
        label_list = docred_df['labels'][doc_id]
        title = docred_df['title'][doc_id]
        for label in label_list:
            entity_h_id = label['h']
            entity_t_id = label['t']
            if title not in ok_entitys_dict:
                ok_entitys_dict[title] = {}
            if entity_h_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_h_id] = {}
            if entity_t_id not in ok_entitys_dict[title][entity_h_id]:
                ok_entitys_dict[title][entity_h_id][entity_t_id] = ""
            ok_entitys_dict[title][entity_h_id][entity_t_id] = "ok"

            if title not in ok_entitys_dict:
                ok_entitys_dict[title] = {}
            if entity_t_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_t_id] = {}
            if entity_h_id not in ok_entitys_dict[title][entity_t_id]:
                ok_entitys_dict[title][entity_t_id][entity_h_id] = ""
            ok_entitys_dict[title][entity_t_id][entity_h_id] = "ok"

else:

    file_path = f"../data/get_entity_pair_selection_label/{data_name}/docred_{data_name}_entity_pair_selection_0-{docred_len}_answer-01-01.jsonl"
    jsonl_data = read_jsonl(file_path)

    for data in jsonl_data:
        entity_h_id = data["h_idx"]
        entity_t_id = data["t_idx"]
        title = data["title"]

        if title not in ok_entitys_dict:
            ok_entitys_dict[title] = {}
        if entity_h_id not in ok_entitys_dict[title]:
            ok_entitys_dict[title][entity_h_id] = {}
        if entity_t_id not in ok_entitys_dict[title][entity_h_id]:
            ok_entitys_dict[title][entity_h_id][entity_t_id] = ""
        ok_entitys_dict[title][entity_h_id][entity_t_id] = "ok"


start = 0

final_list = []

for doc_id in range(start, docred_len):

    title = get_doc_title(doc_id, docred_df)
    doc = get_doc(doc_id, docred_df)
    entitys_list = get_doc_entitys(doc_id, docred_df)

    for index_1 in range(len(entitys_list)):
        for index_2 in range(index_1 + 1, len(entitys_list)):
            for entity_h in entitys_list[index_1]:
                for entity_t in entitys_list[index_2]:
                    entity_h_id = get_entity_id(entity_h, docred_df, doc_id)
                    entity_t_id = get_entity_id(entity_t, docred_df, doc_id)

                    if entity_h_id == entity_t_id:
                        continue

                    if title not in ok_entitys_dict:
                        ok_entitys_dict[title] = {}
                    if entity_h_id not in ok_entitys_dict[title]:
                        ok_entitys_dict[title][entity_h_id] = {}
                    if entity_t_id not in ok_entitys_dict[title][entity_h_id]:
                        ok_entitys_dict[title][entity_h_id][entity_t_id] = ""

                    if ok_entitys_dict[title][entity_h_id][entity_t_id] != "ok":
                        continue


                    prompt = get_prompt_entity_rel(title, doc, entity_h, entity_t)

                    data_dict = {}
                    data_dict["entity_h"] = entity_h
                    data_dict["entity_t"] = entity_t
                    data_dict["entity_h_id"] = entity_h_id
                    data_dict["entity_t_id"] = entity_t_id
                    data_dict["prompt"] = prompt
                    data_dict["entities_description"] = ""
                    data_dict["label_rel"] = []
                    data_dict["title"] = title
                    final_list.append(data_dict)

    print(f"Doc:{title} prompt over")



jsonl_file = f"../data/relation_summary_prompt/{data_name}/result_docred_{data_name}_doc0-{docred_len}.jsonl"

save_to_jsonl(final_list, jsonl_file)
print(f"The result is saved in the file {jsonl_file}")
