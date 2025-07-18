#!/usr/bin/env python3
"""
fine_tune_eng2sql.py  —  Jam-CGPT Spider fine-tuning (robust loader)
"""

import os, json, argparse, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, AdamW
from tqdm.auto import tqdm

# --------------------------------------------------------------------
# Universal Spider loader
# --------------------------------------------------------------------
def read_spider(path: str):
    """Return list[dict(question, query)] no matter the layout."""
    raw = open(path, encoding="utf-8").read().strip()

    # JSON array already?
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # JSON-Lines?
    try:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    except json.JSONDecodeError:
        pass

    # Bare {...}{...}{...}  →  turn into […, …]
    try:
        glued = "[" + raw.replace("}{", "},{") + "]"
        return json.loads(glued)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Cannot parse {path}: {e}") from e

# --------------------------------------------------------------------
class Eng2SQLDataset(Dataset):
    def __init__(self, json_path, tok, eos="<|endoftext|>"):
        data = read_spider(json_path)

        # flatten any nested lists
        def flat(x):
            for item in x:
                if isinstance(item, list):
                    yield from flat(item)
                else:
                    yield item
        data = list(flat(data))

        self.samples = [
            tok.encode(
                d["question"].strip() + eos + d["query"].strip(),
                add_special_tokens=False,
            )
            for d in data
        ]

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return torch.tensor(self.samples[i])

def collate(b):
    return pad_sequence(b, batch_first=True,
                        padding_value=tokenizer.eos_token_id)

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
arg = argparse.ArgumentParser()
arg.add_argument("--pretrained", required=True)
arg.add_argument("--train_json", required=True)
arg.add_argument("--val_json",   required=True)
arg.add_argument("--output_dir", default="out_ft")
arg.add_argument("--epochs", type=int, default=2)
arg.add_argument("--lr",     type=float, default=5e-5)
arg.add_argument("--batch_size", type=int, default=4)
arg.add_argument("--max_len",    type=int, default=256)
arg.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = arg.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# --------------------------------------------------------------------
# Model & tokenizer
# --------------------------------------------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

print("📦 loading Jam-CGPT weights …")
cfg   = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(cfg)
model.load_state_dict(torch.load(args.pretrained, map_location="cpu"), strict=False)
model.resize_token_embeddings(len(tokenizer))
model.to(args.device)

# --------------------------------------------------------------------
# Data
# --------------------------------------------------------------------
train_ds = Eng2SQLDataset(args.train_json, tokenizer)
val_ds   = Eng2SQLDataset(args.val_json,   tokenizer)
train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                      shuffle=True, collate_fn=collate)
val_dl   = DataLoader(val_ds, batch_size=args.batch_size,
                      shuffle=False, collate_fn=collate)

# --------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------
opt = AdamW(model.parameters(), lr=args.lr)

@torch.no_grad()
def val_loss(loader):
    model.eval()
    tot, tok = 0, 0
    for x in loader:
        x = x[:, : args.max_len].to(args.device)
        loss = model(x, labels=x).loss
        tot += loss.item() * x.numel(); tok += x.numel()
    model.train()
    return tot / tok

# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
for ep in range(1, args.epochs + 1):
    pbar = tqdm(train_dl, desc=f"Epoch {ep}")
    for x in pbar:
        x = x[:, : args.max_len].to(args.device)
        loss = model(x, labels=x).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    ppl = torch.exp(torch.tensor(val_loss(val_dl)))
    print(f"Epoch {ep} | val PPL: {ppl:.2f}")

    ckpt = f"{args.output_dir}/ckpt_epoch{ep}.pt"
    torch.save(model.state_dict(), ckpt)
    print("✔ saved", ckpt)

print("✅ done.")
