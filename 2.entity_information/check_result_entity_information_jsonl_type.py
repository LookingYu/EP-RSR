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

file_path = f"../data/entity_information_prompt/{data_name}/prompt_{doc_name}_{data_name}_entity_information_doc0-{docred_len}-type.jsonl"

jsonl_data = read_jsonl(file_path)

len_data = len(jsonl_data)

cnt = 0
start = 0
end = math.ceil(len_data / 200)
print("all doc number:",end)


save_list = []

for jsonl_id in range(start, end):

    jsonl_file_path = f"../data/entity_information_run/{data_name}/result_{doc_name}_{data_name}_entity_information-type_{jsonl_id}.jsonl"
    jsonl_data = read_jsonl(jsonl_file_path)

    for item in jsonl_data:
        if item['response'] == "":
            cnt += 1
            print("-------------------There is an empty response------------------")
        else:
            data_dict = {}
            data_dict["title"] = item["title"]
            data_dict["entity"] = item["entity"]
            data_dict["prompt"] = item["prompt"]
            data_dict["response"] = item["response"]
            save_list.append(data_dict)


save_path = f"../data/entity_information/{data_name}/result_{doc_name}_{data_name}_entity_information_0-{docred_len}-type-nodeal.jsonl"
save_to_jsonl(save_list, save_path)
print(f"The result is saved in the file {save_path}")
print(f"There are {cnt} empty data records")

