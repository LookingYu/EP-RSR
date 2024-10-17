## Model Deployment

### Step 1: Set up Environment

#### Create llama3 Environment

```
conda create -n llama3 python=3.10 
```

#### View Current Environments

```
conda env list
```

#### Activate Target Virtual Environment (llama3)

```
conda activate llama3
```

#### Install Environment Dependencies

```
pip install -r requirements.txt
```

### Step 2: Upload Model

```
../model-path/
```

### Step 3: Deploy Model

```
Run ../0.pre_model/llama3-api.py, model deployed at host='127.0.0.1', port=6006
```

## Data Preprocessing

### 1. Construct Training Data

#### Obtain Train Entity Information

```
Run the following files in order to get entity information jsonl 
(parameter data_name is: train_annotated)
../1.entity_information/entity_information_prompt.py 
../1.entity_information/entity_information_run.py  
../1.entity_information/check_result_entity_information_jsonl.py 
```

#### Obtain Train Relation Summary

```
Run the following files in order to get relation summary jsonl 
(parameter data_name is: train_annotated)
../3.relation_summary/relation_summary_prompt.py
../3.relation_summary/relation_summary_run.py
../3.relation_summary/check_result_relation_summary_jsonl.py
```

#### Obtain Train Encoding Representation

```
Run the following file in order to get encoding representation jsonl 
(parameter data_name is: train_annotated)
../4.retrieval/get_embeddings.py
```

### 2. Construct Fine-tuning Data

#### Obtain Entity Pair Selection Fine-tuning Data

```
Run the following file in order to get fine-tuning data 
(parameter data_name is: train_annotated)
../finetuning/get_data_entity_pair_selection.py
```

#### Obtain Multiple Choice Fine-tuning Data

```
Run the following files in order to get fine-tuning data 
(parameter data_name is: train_annotated)
../4.retrieval/retrieval_from_train-few.py
../5.multiple_choice/multiple_choice_prompt.py
../finetuning/get_data_multiple_choice.py
```

#### Obtain Triplet Fact Judgement Fine-tuning Data

```
Run the following files in order to get fine-tuning data 
(parameter data_name is: train_annotated)
../6.triplet_fact_judgement/triplet_fact_judgement_prompt.py
../finetuning/get_data_triplet_fact_judgement.py
```

## Main Run

### I. Obtain Dev Entity Information

```
Run the following files in order to get entity information jsonl 
(parameter data_name is: dev)
../1.entity_information/entity_information_prompt.py 
../1.entity_information/entity_information_run.py  
../1.entity_information/check_result_entity_information_jsonl.py 
```

### II. Obtain Selected Entity Pairs for Dev

```
Run the following files in order to get selected entity pairs jsonl 
(parameter data_name is: dev)
../2.entity_pair_selection/entity_pair_selection_prompt.py
../2.entity_pair_selection/entity_pair_selection_run.py
../2.entity_pair_selection/check_result_entity_pair_selection_jsonl.py
../2.entity_pair_selection/get_entity_pair_selection_label.py
```

### III. Obtain Dev Relation Summary

```
Run the following files in order to get relation summary jsonl 
(parameter data_name is: dev)
../3.relation_summary/relation_summary_prompt.py
../3.relation_summary/relation_summary_run.py
../3.relation_summary/check_result_relation_summary_jsonl.py
```

### IV. Retrieve Coarsely Filtered Candidate Relations

```
Run the following files in order to get the path of candidate relations after coarse filtering 
(parameter data_name is: dev)
../4.retrieval/get_embeddings.py
../4.retrieval/retrieval_from_train-few.py
```

### V. Obtain Fine-filtered Candidate Relations

```
Run the following files in order to get fine-filtered candidate relations jsonl 
(parameter data_name is: dev)
../5.multiple_choice/multiple_choice_prompt.py
../5.multiple_choice/multiple_choice_run.py
../5.multiple_choice/check_result_multiple_choice_jsonl.py
../5.multiple_choice/get_multiple_choice_label.py
```

### VI. Obtain Triplet Facts

```
Run the following files in order to get triplet facts jsonl 
(parameter data_name is: dev)
../6.triplet_fact_judgement/triplet_fact_judgement_prompt.py
../6.triplet_fact_judgement/triplet_fact_judgement_run.py
../6.triplet_fact_judgement/check_result_triplet_fact_judgement_jsonl.py
../6.triplet_fact_judgement/get_triplet_fact_judgement_label.py
```
