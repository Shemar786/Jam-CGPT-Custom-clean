import torch
import tiktoken
from model import GPTConfig, GPT

# Load model
ckpt_path = 'out-jam-cgpt/ckpt_len1024.pt'
checkpoint = torch.load(ckpt_path, map_location='cuda')
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'], strict=False)
model.eval().to('cuda')

# Encoding
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Preprocess input like training examples
java_method = """public int multiply(int a, int b) {
    return a * b;
}"""
java_tokens = java_method.replace("\n", "<NL>")
prompt = f"TDAT: {java_tokens} COM:"

# Encode and run generation
x = torch.tensor(encode(prompt), dtype=torch.long, device='cuda')[None, ...]
with torch.no_grad():
    y = model.generate(x, max_new_tokens=50, temperature=0.1, top_k=50)
    out = decode(y[0].tolist())

# Extract summary
if "COM:" in out:
    result = out.split("COM:")[1].split("<|endoftext|>")[0]
else:
    result = out

print("\n🔍 Generated Summary:\n", result.strip())
