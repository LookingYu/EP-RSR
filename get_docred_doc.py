import pandas as pd
import numpy as np

def get_doc(doc_id, df):


    sentence_str = ""
    for sentence in df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "
    return sentence_str
