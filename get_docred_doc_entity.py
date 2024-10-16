import pandas as pd
import numpy as np

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




