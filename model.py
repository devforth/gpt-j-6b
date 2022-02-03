from transformers import GPTJForCausalLM, AutoConfig, GPT2Tokenizer, AutoTokenizer
import torch

print('calling .from_pretrained start')
model = GPTJForCausalLM.from_pretrained("./gpt-j-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
print('.from_pretrained end')

# tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
print('AutoTokenizer.from_pretrained end')


# model.half().cuda()
# model.half()

input_text = "Hello my name is Paul and"
input_ids = tokenizer.encode(str(input_text), return_tensors='pt')

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=20,
    top_p=0.7,
    top_k=0,
    temperature=1.0,
)

print('output', tokenizer.decode(output[0], skip_special_tokens=True))

def eval(input):
    input_ids = tokenizer.encode(str(input["text"]), return_tensors='pt').cuda()
    output = model.generate(
        input_ids,
        do_sample=True,
        max_length=input["length"],
        top_p=input["top_p"],
        top_k=input["top_k"],
        temperature=input["temperature"],
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
