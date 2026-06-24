import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict
import transformers
import re
from PIL import Image

disable_torch_init()
model_path = 'neulab/Pangea-7B'
model_name = 'Pangea-7B-qwen'
tokenizer_text_only, model_text_only, _, context_len_text_only = load_pretrained_model(model_path, None, model_name)
args = {"multimodal": True}
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, **args)

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
    input_ids = []
    source = sources
    if roles[source[0]["from"]] != roles["human"]: source = source[1:]
    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1: _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None: _input_id = tokenizer(role).input_ids + nl_tokens
            else: _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
    input_ids.append(input_id)
    return torch.tensor(input_ids, dtype=torch.long)

def generate_output_text_only(prompt, do_sample=False, temperature=0, top_p=0.5, num_beams=1, max_new_tokens=1024):
    input_ids = preprocess_qwen([{'from': 'human', 'value': prompt},{'from': 'gpt','value': None}], tokenizer_text_only, has_image=False).cuda()
    with torch.inference_mode():
        generated_ids = model_text_only.generate(
            input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]
    outputs = tokenizer_text_only.batch_decode(generated_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs

def generate_output(prompt, image=None, do_sample=False, temperature=0, top_p=0.5, num_beams=1, max_new_tokens=1024):
    if image == None: return generate_output_text_only(prompt, do_sample, temperature, top_p, num_beams, max_new_tokens)
    image_tensors = []
    prompt = "<image>\n" + prompt
    image = Image.open(image)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensors.append(image_tensor.half().cuda())
    input_ids = preprocess_qwen([{'from': 'human', 'value': prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
    print(input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs

# image + text
prompt = "What did you see in the image"
image = "image.png"
print(generate_output(prompt, image=image))

# text-only
prompt = "Write me a python function that could sort a input integer list by descending order"
print(generate_output(prompt))
