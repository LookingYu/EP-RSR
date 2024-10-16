import pandas as pd
import numpy as np
import torch


def get_prompt_entity(title, doc, entity):
    prompt = f"""The text is as follows:

{title}
{doc}

What is "{entity}"? (For example, US is a country and 12 is a number) Answer in one sentence.Only output answers without outputting anything else.
The answer is:"""
    return prompt

def get_prompt_entity_rel(title,doc,entity_h,entity_t):
    prompt = f"""The text is as follows:

{title}
{doc}

What is the relationship between {entity_h} and {entity_t}?
Answer in one sentence.Only output answers without outputting anything else."""
    return prompt
