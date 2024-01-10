import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "kyujinpy/Ko-PlatYi-6B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="cuda:0")

device = "cuda:0"

# https://huggingface.co/docs/transformers/chat_templating
messages = [
    {"role": "user", "content": "한국의 서울대학교에 대해 설명해줘"}
]


encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)


generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
output = tokenizer.batch_decode(generated_ids)[0]
print('output:', output)