from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device_map = {"": 0}

##Base Model Path
model_name_or_path = "../llama3-8b-instruct"
##Adapter path
adapter_name_or_path = "output/llame3-8b-instruct-llama-factory"
## Merge model path
save_path = "new_merge_model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device_map)
print("Successfully loaded the original model")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("Successfully loaded the fine-tuned model")
model = model.merge_and_unload()
print("Successfully merged the models")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Model saved successfully")




