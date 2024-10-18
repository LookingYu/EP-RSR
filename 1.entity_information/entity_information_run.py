import subprocess
import json
import pandas as pd
import numpy as np
import json
import requests
import re
import csv

def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_one(prompt):

    prompt_list = []
    prompt_list.append(prompt)
    system_prompt = ""
    message = prompt_list
    temperature = 0.9
    max_new_tokens = 50
    data = {}
    data["system_prompt"] = system_prompt
    data["message"] = message
    data["temperature"] = temperature
    data["max_new_tokens"] = max_new_tokens

    local_file = "local_prompt.json"
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    remote_path = "local_prompt.json"

    final_command = f'curl -X POST "http://127.0.0.1:6006" -H \'Content-Type: application/json\' -d@\'{remote_path}\' '

    result = subprocess.run(final_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result_list = result.stdout

    if result_list == "Internal Server Error":
        return ""

    else:

        result_list = eval(result_list)
        response = result_list[0]

        assistant_start_pos = response.find("<|start_header_id|>assistant<|end_header_id|>")
        result_response_1 = response[assistant_start_pos:]

        assistant_end_pos = result_response_1.find("<|eot_id|>")
        result_response_2 = result_response_1[:assistant_end_pos]

        result_response = result_response_2.replace("<|start_header_id|>assistant<|end_header_id|>", "", 1)

        result_response_unique = result_response.strip()

        return result_response_unique

def run_list(prompt_list):
    system_prompt = ""
    message = prompt_list
    temperature = 0.9
    max_new_tokens = 50

    data = {}
    data["system_prompt"] = system_prompt
    data["message"] = message
    data["temperature"] = temperature
    data["max_new_tokens"] = max_new_tokens

    local_file = "local_prompt.json"
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    remote_path = "local_prompt.json"

    final_command = f'curl -X POST "http://127.0.0.1:6006" -H \'Content-Type: application/json\' -d@\'{remote_path}\' '

    result = subprocess.run(final_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result_list = result.stdout

    response_list_run = []

    if result_list == "Internal Server Error":
        for id in range(len(prompt_list)):
            prompt = prompt_list[id]
            response = run_one(prompt)
            response_list_run.append(response)

    else:

        result_list = eval(result_list)

        for response in result_list:


            assistant_start_pos = response.find("<|start_header_id|>assistant<|end_header_id|>")
            result_response_1 = response[assistant_start_pos:]

            assistant_end_pos = result_response_1.find("<|eot_id|>")
            result_response_2 = result_response_1[:assistant_end_pos]

            result_response = result_response_2.replace("<|start_header_id|>assistant<|end_header_id|>", "", 1)

            result_response_unique = result_response.strip()

            response_list_run.append(result_response_unique)

    return response_list_run


#------------------------------------------------------------------------------------------
data_name = "dev" # test dev train_annotated
#------------------------------------------------------------------------------------------

doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"

fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)


file_path = f"../data/entity_information_prompt/{data_name}/prompt_docred_{data_name}_entity_information_doc0-{docred_len}.jsonl"
jsonl_data = read_jsonl(file_path)

len_data = len(jsonl_data)

print("data len: ",len_data)
print("----------------------------------")



batch_size = 20
prompt_list = []
response_list = []
id_list = []

start = 0
save_id = 0
end = len_data
save_data_list = []
save_cnt = 0


for id in range(start, end):

    if len(prompt_list) == batch_size:

        response_list = run_list(prompt_list)

        for i in range(batch_size):
            now_id = id_list[i]
            if i >= len(response_list):
                jsonl_data[now_id]["response"] = []
                print(f"{now_id} data query failed")
            else:
                response = response_list[i]
                jsonl_data[now_id]["response"] = response
                print(f"{now_id} data query completed")

            save_data_list.append(jsonl_data[now_id])

        prompt_list.clear()
        id_list.clear()

    if save_cnt == 200:

        save_name = f"../data/entity_information_run/{data_name}/result_docred_{data_name}_entity_information_{save_id}.jsonl"
        save_to_jsonl(save_data_list, save_name)
        print(f"The result is saved in the file {save_name}")
        save_id += 1
        save_cnt = 0
        save_data_list.clear()

    prompt = jsonl_data[id]["prompt"]
    prompt_list.append(prompt)
    id_list.append(id)
    save_cnt += 1




if len(prompt_list) > 0:

    response_list = run_list(prompt_list)

    for i in range(len(id_list)):
        response = response_list[i]
        now_id = id_list[i]
        jsonl_data[now_id]["response"] = response
        print(f"{now_id} data query completed")

        save_data_list.append(jsonl_data[now_id])

    prompt_list.clear()
    id_list.clear()

    save_name = f"../data/entity_information_run/{data_name}/result_docred_{data_name}_entity_information_{save_id}.jsonl"
    save_to_jsonl(save_data_list, save_name)
    print(f"The result is saved in the file {save_name}")
    save_id += 1
    save_cnt = 0
    save_data_list.clear()



