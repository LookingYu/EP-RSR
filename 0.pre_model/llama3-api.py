import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import uvicorn, json, datetime
from transformers import LlamaForCausalLM
import torch

device_map = {"": 0}

DEVICE = "cuda"
DEVICE_ID = "0"
# CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
CUDA_DEVICE = "cuda:0"

device = torch.device("cuda:0")

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()



@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    system_prompt = json_post_list.get('system_prompt')
    message = json_post_list.get('message')
    temperature = json_post_list.get('temperature')
    max_new_tokens = json_post_list.get('max_new_tokens')


    inputs_list = []
    for prompt in message:
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,return_tensors='pt')
        inputs_list.append(inputs)



    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = "[PAD]"

    inputs = tokenizer(inputs_list, return_tensors="pt", padding=True).to(device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|eot_id|>"])]

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators[0],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        attention_mask = inputs.attention_mask
    ).to(device)

    response_list = []
    for seq in outputs:
        response = tokenizer.decode(seq, skip_special_tokens=False)
        response_list.append(response)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(time)
    print(response_list)

    torch_gc()
    return response_list


if __name__ == '__main__':

    # model = LlamaForCausalLM.from_pretrained("../../llama3-8b-instruct/", load_in_8bit=False, load_in_4bit=False,
    #                                          torch_dtype=torch.float16, device_map=device_map)
    # tokenizer = AutoTokenizer.from_pretrained("../../llama3-8b-instruct/", device_map=device_map)

    model = LlamaForCausalLM.from_pretrained("../finetuning/new_merge_model/", load_in_8bit=False, load_in_4bit=False,
                                             torch_dtype=torch.float16, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained("../finetuning/new_merge_model/", device_map=device_map)

    # model = model.bfloat16()
    model.eval()
    uvicorn.run(app, host='127.0.0.1', port=6006, workers=1)
