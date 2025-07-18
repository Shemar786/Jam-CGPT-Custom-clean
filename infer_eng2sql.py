import os
import sys
sys.path.insert(0, os.path.abspath(os.getcwd()))

import torch
from mingpt.model import GPT, GPTConfig
from tiktoken import get_encoding

# 1. Load checkpoint
try:
    ckpt = torch.load('out_eng2sql_350m/ckpt.pt', map_location='cpu')
except FileNotFoundError:
    print("Error: Checkpoint 'out_eng2sql_350m/ckpt.pt' not found")
    sys.exit(1)

config = GPTConfig(**ckpt['model_args'])
model = GPT(config)
model.load_state_dict(ckpt['model'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval().to(device)

# 2. Prepare tokenizer
enc = get_encoding('gpt2')
try:
    eos_id = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
except IndexError:
    print("Error: Failed to encode end-of-text token")
    sys.exit(1)

# 3. Encode your English question
question = "List the name of all students older than 20 in the student table."
input_ids = enc.encode(question) + [eos_id]
x = torch.tensor([input_ids], dtype=torch.long).to(device)

# 4. Generate SQL
with torch.no_grad():
    y = model.generate(
        x,
        max_new_tokens=128,
        temperature=0.7,
        top_k=50
    )

# 5. Decode and print
generated = enc.decode(y[0].tolist())
print("Generated SQL:", generated)
