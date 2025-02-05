import json
import math
import pandas as pd

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')


data_name = "dev"
doc_name = "redocred"

doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}_revised.json"

fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

file_path = f"../data/entity_pair_selection_prompt/{data_name}/entity_pair_selection_prompt_{data_name}_01_RTE_{doc_name}.jsonl"

jsonl_data = read_jsonl(file_path)

len_data = len(jsonl_data)

cnt = 0
start = 0
end = math.ceil(len_data / 200)
print("all doc number:",end)

save_doc_name = "01-RTE"

save_list = []

for jsonl_id in range(start, end):

    jsonl_file_path = f"../data/entity_pair_selection_run/{data_name}/result_{doc_name}_{data_name}_entity_pair_selection-{save_doc_name}_{jsonl_id}.jsonl"
    jsonl_data = read_jsonl(jsonl_file_path)

    for item in jsonl_data:
        if item['response'] == "":
            cnt += 1
            print("-------------------There is an empty response------------------")
        else:
            data_dict = {}
            data_dict["title"] = item["title"]
            data_dict["doc_id"] = item["doc_id"]
            data_dict["prompt"] = item["prompt"]
            data_dict["instruction"] = item["instruction"]
            data_dict["input"] = item["input"]
            data_dict["response"] = item["response"]
            save_list.append(data_dict)

save_path = f"../data/check_result_entity_pair_selection_jsonl/{data_name}/result_{doc_name}_{data_name}_entity_pair_selection_0-{docred_len}-{save_doc_name}.jsonl"

save_to_jsonl(save_list, save_path)
print(f"The result is saved in the file {save_path}")
print(f"There are {cnt} empty data records")

