import json
import torch
import pandas as pd


data_name = "dev"  #train_annotated
doc_name =  "redocred"

doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}_revised.json"
fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)

info_fr = open(f'../data/{doc_name}/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)
reverse_rel_info = {v: k for k, v in rel_info.items()}

print(f"{doc_name}")
print(f"{data_name}")
print(f"doc num:{docred_len}")


for doc_id in range(docred_len):
    title = docred_df["title"][doc_id]
    print(title)

    sentence_str = ""
    for sentence in docred_df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "

    print(sentence_str)
    print("--------------------")

    label_list = docred_df["labels"][doc_id]
    for label in label_list:
        name_h = docred_df['vertexSet'][doc_id][label['h']][0]['name']
        name_t = docred_df['vertexSet'][doc_id][label['t']][0]['name']
        print(label['h'], rel_info[label['r']], label['t'])
        print(name_h, rel_info[label['r']], name_t)
        print("-----------------")


