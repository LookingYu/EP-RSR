from get_docred_doc_entity import get_doc_entitys
from get_docred_doc import get_doc
from get_docred_doc_title import get_doc_title
from entity_judge import get_entity_id
from get_prompt import get_prompt_entity_rel, get_prompt_entity
import pandas as pd
import json


def save_to_jsonl(data, jsonl_file):
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii=False)
            jsonlfile.write('\n')


data_name = "dev"

doc_dir = '../data/docred/'
doc_filename = f"{doc_dir}{data_name}.json"
fr = open(doc_filename, 'r', encoding='utf-8')
json_info = fr.read()
docred_df = pd.read_json(json_info)
docred_len = len(docred_df)


start = 0
end = docred_len


final_list = []
for doc_id in range(start, end):

    title = get_doc_title(doc_id, docred_df)

    doc = get_doc(doc_id, docred_df)

    entitys_list = get_doc_entitys(doc_id, docred_df)
    for index_1 in range(len(entitys_list)):
        for entity in entitys_list[index_1]:

            prompt = get_prompt_entity(title, doc, entity)

            entity_id = get_entity_id(entity, docred_df, doc_id)

            data_dict = {}
            data_dict["title"] = title
            data_dict["entity"] = entity
            data_dict["entity_id"] = entity_id
            data_dict["prompt"] = prompt
            data_dict["response"] = ""
            final_list.append(data_dict)

    print(f"Doc:{title} prompt over")


jsonl_file = f"../data/entity_information_prompt/{data_name}/prompt_docred_{data_name}_entity_information_doc0-{docred_len}.jsonl"

save_to_jsonl(final_list, jsonl_file)
print(f"The result is saved in the file {jsonl_file}")
