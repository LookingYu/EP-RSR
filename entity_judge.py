import pandas as pd
import numpy as np

def get_entity_id(entity, df, doc_id):

    len_doc = len(df['vertexSet'][doc_id])

    for entity_id in range(len_doc):
        for entity_name in df['vertexSet'][doc_id][entity_id]:
            if entity_name['name'] == entity:
                return entity_id

    return -1
