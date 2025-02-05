import pandas as pd
import json

def get_prompt_entity_rel(title,doc,entity_h,entity_t):
    prompt = f"""The text is as follows:

{title}
{doc}

What is the relationship between {entity_h} and {entity_t}?
Answer in one sentence.Only output answers without outputting anything else."""
    return prompt


def get_doc_title(doc_id, df):

    title = df['title'][doc_id]

    return title


def get_doc(doc_id, df):

    sentence_str = ""
    for sentence in df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "
    return sentence_str


def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii=False)
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

    file_path = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}_answer-01-RTE.jsonl"
    jsonl_data = read_jsonl(file_path)
    file_path_02 = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}_answer-01-02-RTE.jsonl"
    jsonl_data_02 = read_jsonl(file_path_02)
    file_path_03 = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}_answer-01-03-RTE.jsonl"
    jsonl_data_03 = read_jsonl(file_path_03)
    file_path_04 = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}_answer-01-04-RTE.jsonl"
    jsonl_data_04 = read_jsonl(file_path_04)
    file_path_05 = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}_answer-01-05-RTE.jsonl"
    jsonl_data_05 = read_jsonl(file_path_05)

    for data in jsonl_data:
        entity_h_name = data["h_name"]
        entity_t_name = data["t_name"]
        title = data["title"]

        if title not in ok_entitys_dict:
            ok_entitys_dict[title] = {}
        if entity_h_name not in ok_entitys_dict[title]:
            ok_entitys_dict[title][entity_h_name] = {}
        if entity_t_name not in ok_entitys_dict[title][entity_h_name]:
            ok_entitys_dict[title][entity_h_name][entity_t_name] = ""
        ok_entitys_dict[title][entity_h_name][entity_t_name] = "ok"

    for data in jsonl_data_02:
        entity_h_name = data["h_name"]
        entity_t_name = data["t_name"]
        title = data["title"]

        if title not in ok_entitys_dict:
            ok_entitys_dict[title] = {}
        if entity_h_name not in ok_entitys_dict[title]:
            ok_entitys_dict[title][entity_h_name] = {}
        if entity_t_name not in ok_entitys_dict[title][entity_h_name]:
            ok_entitys_dict[title][entity_h_name][entity_t_name] = ""
        ok_entitys_dict[title][entity_h_name][entity_t_name] = "ok"

    for data in jsonl_data_03:
        entity_h_name = data["h_name"]
        entity_t_name = data["t_name"]
        title = data["title"]

        if title not in ok_entitys_dict:
            ok_entitys_dict[title] = {}
        if entity_h_name not in ok_entitys_dict[title]:
            ok_entitys_dict[title][entity_h_name] = {}
        if entity_t_name not in ok_entitys_dict[title][entity_h_name]:
            ok_entitys_dict[title][entity_h_name][entity_t_name] = ""
        ok_entitys_dict[title][entity_h_name][entity_t_name] = "ok"

    for data in jsonl_data_04:
        entity_h_name = data["h_name"]
        entity_t_name = data["t_name"]
        title = data["title"]

        if title not in ok_entitys_dict:
            ok_entitys_dict[title] = {}
        if entity_h_name not in ok_entitys_dict[title]:
            ok_entitys_dict[title][entity_h_name] = {}
        if entity_t_name not in ok_entitys_dict[title][entity_h_name]:
            ok_entitys_dict[title][entity_h_name][entity_t_name] = ""
        ok_entitys_dict[title][entity_h_name][entity_t_name] = "ok"

    for data in jsonl_data_05:
        entity_h_name = data["h_name"]
        entity_t_name = data["t_name"]
        title = data["title"]

        if title not in ok_entitys_dict:
            ok_entitys_dict[title] = {}
        if entity_h_name not in ok_entitys_dict[title]:
            ok_entitys_dict[title][entity_h_name] = {}
        if entity_t_name not in ok_entitys_dict[title][entity_h_name]:
            ok_entitys_dict[title][entity_h_name][entity_t_name] = ""
        ok_entitys_dict[title][entity_h_name][entity_t_name] = "ok"


start = 0

final_list = []

for doc_id in range(start, docred_len):

    title = get_doc_title(doc_id, docred_df)
    doc = get_doc(doc_id, docred_df)

    if title not in ok_entitys_dict:
        continue

    entitys_list = []
    for key, value in ok_entitys_dict[title].items():
        entity_h_name = key
        if entity_h_name not in entitys_list:
            entitys_list.append(entity_h_name)

    for index_1 in range(len(entitys_list)):
        for index_2 in range(index_1 + 1, len(entitys_list)):
            entity_h = entitys_list[index_1]
            entity_t = entitys_list[index_2]

            if title not in ok_entitys_dict:
                ok_entitys_dict[title] = {}
            if entity_h not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_h] = {}
            if entity_t not in ok_entitys_dict[title][entity_h]:
                ok_entitys_dict[title][entity_h][entity_t] = ""

            if ok_entitys_dict[title][entity_h][entity_t] != "ok":
                continue


            prompt = get_prompt_entity_rel(title, doc, entity_h, entity_t)

            data_dict = {}
            data_dict["entity_h"] = entity_h
            data_dict["entity_t"] = entity_t
            data_dict["prompt"] = prompt
            data_dict["entities_description"] = ""
            data_dict["label_rel"] = []
            data_dict["title"] = title
            final_list.append(data_dict)

    print(f"Doc:{title} prompt over")



jsonl_file = f"../data/relation_summary_prompt/{data_name}/result_{doc_name}_{data_name}_doc0-{docred_len}-RTE.jsonl"

save_to_jsonl(final_list, jsonl_file)
print(f"The result is saved in the file {jsonl_file}")
print(len(final_list))