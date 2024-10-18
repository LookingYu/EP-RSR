import json
import pandas as pd
import math

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii=False)
            jsonlfile.write('\n')
def label_judge(entity_h_id, entity_t_id, df, doc_id, rel_info):
    rels = []
    for relation_dict in df['labels'][doc_id]:
        if relation_dict['h'] == entity_h_id and relation_dict['t'] == entity_t_id:
            rels.append(rel_info[relation_dict['r']])
        if relation_dict['h'] == entity_t_id and relation_dict['t'] == entity_h_id:
            rels.append(rel_info[relation_dict['r']])

    unique_rels = list(set(rels))
    return unique_rels

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id


data_name = "dev"

doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_fr = open(doc_filename, 'r', encoding='utf-8')
json_info = docred_fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)


file_path = f"../data/relation_summary_prompt/{data_name}/result_docred_{data_name}_doc0-{docred_len}.jsonl"

jsonl_data = read_jsonl(file_path)
len_data = len(jsonl_data)


start = 0
end = math.ceil(len_data / 200)
print("all doc number:",end)

save_list = []
cnt = 0

info_fr = open('../data/docred/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)



for jsonl_id in range(start, end):

    jsonl_file_path = f"../data/relation_summary_run/{data_name}/result_docred_{data_name}_relation_summary_{jsonl_id}.jsonl"
    jsonl_data = read_jsonl(jsonl_file_path)

    for item in jsonl_data:
        if item['entities_description'] == "":
            cnt += 1
            print("-------------------There is an empty response------------------")
        else:
            data_dict = {}
            data_dict["entity_h"] = item["entity_h"]
            data_dict["entity_t"] = item["entity_t"]
            data_dict["entity_h_id"] = item["entity_h_id"]
            data_dict["entity_t_id"] = item["entity_t_id"]
            data_dict["entities_description"] = item["entities_description"]
            data_dict["title"] = item["title"]

            if data_name == "dev" or data_name == "train":
                doc_id = get_docid(item["title"], docred_df)

                unique_rels = label_judge(item["entity_h_id"], item["entity_t_id"], docred_df, doc_id, rel_info)
                if len(unique_rels) == 0:
                    data_dict["label_rel"] = ["no_relation"]
                else:
                    data_dict["label_rel"] = unique_rels

            save_list.append(data_dict)



save_path = f"../data/check_result_relation_summary_jsonl/{data_name}/result_docred_{data_name}_relation_summary_0-{docred_len}.jsonl"
save_to_jsonl(save_list, save_path)
print(f"The result is saved in the file {save_path}")
print(f"There are {cnt} empty data records")
