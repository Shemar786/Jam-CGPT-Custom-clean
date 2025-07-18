# quick_test.py
import torch, pickle
from model import GPT, GPTConfig
from myutils import sample   # the sampling helper from sample_jam_cgpt.py

# 1) Load your fine-tuned checkpoint
ckpt = torch.load('out-jam-cgpt/ckpt_retrain.pt', map_location='cpu')
config = GPTConfig(**ckpt['model_args'])
model  = GPT(config)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 2) A helper to generate text
def generate(prompt, max_new=64, temp=0.7, top_k=200):
    # sample() takes (model, prompt, max_new, temperature, top_k)
    return sample(model, prompt, max_new, temperature=temp, top_k=top_k)

# 3) Run a few test prompts
tests = [
    "SELECT name FROM employees WHERE department = '",
    "INSERT INTO orders (user_id, total) VALUES (",
]
for prompt in tests:
    out = generate(prompt)
    print(f"\nPROMPT: {prompt!r}\nCOMPLETION: {out!r}\n")
