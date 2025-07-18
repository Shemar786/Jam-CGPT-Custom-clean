import torch
import tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

# ---- Config ----
device          = 'cuda'
out_dir         = 'out-jam-cgpt'
ckpt_file       = 'ckpt_len1024.pt'
dtype           = 'float16'
temperature     = 0.8
top_k           = 50
max_new_tokens  = 40

# ---- Your Java method ----
java_method = """public boolean isEven(int n) {
    return n % 2 == 0;
}"""

prompt = f"TDAT:\n{java_method}\nCOM:"

# ---- Load model ----
checkpoint = torch.load(f'{out_dir}/{ckpt_file}', map_location=device)
gptconf    = GPTConfig(**checkpoint['model_args'])
model      = GPT(gptconf)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval().to(device)

# ---- Tokenizer (GPT-2 BPE) ----
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda ids: enc.decode(ids)

# ---- Sampling context ----
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx     = nullcontext() if device=='cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# ---- Run inference ----
tokens = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
with torch.no_grad():
    with ctx:
        # note: second arg is positional here
        y = model.generate(tokens, max_new_tokens, temperature=temperature, top_k=top_k)
output = decode(y[0].tolist())

# ---- Extract and print the COM: summary ----
if 'COM:' in output:
    summary = output.split('COM:')[1].split('<|endoftext|>')[0].strip()
else:
    summary = output.strip()

print("=== Model's Summary ===")
print(summary)
