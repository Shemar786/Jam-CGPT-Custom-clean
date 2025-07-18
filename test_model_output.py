import torch, pickle
from types import SimpleNamespace
from model import GPT

# ---------- 1. Load checkpoint ----------
ckpt_path = "out-so13m-finetuned/ckpt_sql8k_so13m_finetuned.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# ---------- 2. Build config object ----------
cfg = SimpleNamespace(**ckpt["model_args"])
# provide any attributes __init__ expects but aren’t in the dict
cfg.lora_dropout = 0.0
cfg.attn_pdrop   = getattr(cfg, "attn_pdrop", cfg.dropout)
cfg.resid_pdrop  = getattr(cfg, "resid_pdrop", cfg.dropout)

# ---------- 3. Strip `_orig_mod.` from state_dict ----------
clean_state = {}
for k, v in ckpt["model"].items():
    clean_state[k.replace("_orig_mod.", "")] = v

# ---------- 4. Re‑create model & load weights ----------
model = GPT(cfg)
model.load_state_dict(clean_state, strict=False)
model.eval()

# ---------- 5. Load tokenizer ----------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

prompt = "Which cities have more than 3 airports?"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

with torch.no_grad():
    out_ids = model.generate(input_ids, max_new_tokens=100)

print("\nPrompt :", prompt)
print("SQL    :", tokenizer.decode(out_ids[0].tolist()))
