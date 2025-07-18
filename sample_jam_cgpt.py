#!/usr/bin/env python3
"""
sample_jam_cgpt.py
==================

• SINGLE‑PROMPT MODE
    python3 sample_jam_cgpt.py config/finetune_small_sql8k.py \
        --out_dir out-sql8k-scratch \
        --outfilename ckpt_sql8k_scratch.pt \
        --prompt "Show the status …"
  → prints ONLY the generated SQL.

• BATCH MODE (omit --prompt)
  Acts like the original script: samples N rows from a CSV and writes
  jam_cgpt_predictions/<prediction_filename>.
"""

import os, csv, argparse, torch, tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

# ── Argument parsing ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sample Jam‑CGPT model")
parser.add_argument("config_file", help="Path to the model‑config Python file")

# Shared flags
parser.add_argument("--out_dir",     required=True)
parser.add_argument("--outfilename", required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype",  choices=["float32","bfloat16","float16"],
                    default="float16")

# Single‑prompt flag
parser.add_argument("--prompt",
                    help="ONE English question → prints SQL and exits")

# Batch‑mode‑only flags
parser.add_argument("--batch_file")
parser.add_argument("--prediction_filename")
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--temperature",   type=float, default=0.2)
parser.add_argument("--top_k",         type=int,   default=200)
args = parser.parse_args()

# ── Sanity check for batch mode ───────────────────────────────
if not args.prompt:
    missing = [x for x in ("batch_file", "prediction_filename")
               if getattr(args, x) is None]
    if missing:
        parser.error("--" + " and --".join(missing) +
                     " required when --prompt is not used")

# ── Load finetune config (exec pattern) ───────────────────────
cfg = {}
with open(args.config_file, "r") as f:
    exec(f.read(), cfg)

# ── Load checkpoint & build model ─────────────────────────────
ckpt_path  = os.path.join(args.out_dir, args.outfilename)
ckpt       = torch.load(ckpt_path, map_location=args.device)
model_cfg  = GPTConfig(**ckpt["model_args"])
model      = GPT(model_cfg)

# strip '_orig_mod.' prefixes
state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
         for k, v in ckpt.get("model", ckpt).items()}
model.load_state_dict(state, strict=True)

device_type = "cuda" if args.device.startswith("cuda") else "cpu"
model.to(args.device)
model.eval()
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type, dtype=getattr(torch, args.dtype))

# ── Tokenizer helpers ─────────────────────────────────────────
enc    = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda ids: enc.decode(ids)

def translate(question: str,
              max_new: int = 64,
              temp: float = 0.2,
              top_k: int = 200) -> str:
    prompt = f"ENG:\t{question}\nSQL:\t"
    ids    = encode(prompt)
    x      = torch.tensor([ids], dtype=torch.long, device=args.device)

    with torch.no_grad(), ctx:
        y = model.generate(x, max_new_tokens=max_new,
                           temperature=temp, top_k=top_k)

    gen = y[0].tolist()[len(ids):]
    if enc.eot_token in gen:
        gen = gen[:gen.index(enc.eot_token)]
    return decode(gen).strip()

# ── SINGLE‑PROMPT MODE ────────────────────────────────────────
if args.prompt:
    sql = translate(args.prompt.strip(),
                    max_new=args.max_new_tokens,
                    temp=args.temperature,
                    top_k=args.top_k)

    # strip "SQL:" prefix and <|endoftext|> tail, if any
    if "SQL:" in sql:
        sql = sql.split("SQL:", 1)[-1].strip()
    if "<|endoftext|>" in sql:
        sql = sql.split("<|endoftext|>", 1)[0].strip()

    print(sql)
    raise SystemExit(0)

# ── BATCH MODE ────────────────────────────────────────────────
with open(args.batch_file, newline='', encoding="utf-8") as f:
    prompts = [row["text_query"] for row in csv.DictReader(f)]
prompts = prompts[:args.num_samples]

os.makedirs("jam_cgpt_predictions", exist_ok=True)
out_path = os.path.join("jam_cgpt_predictions", args.prediction_filename)

with open(out_path, "w", encoding="utf-8") as outf:
    for q in prompts:
        sql = translate(q,
                        max_new=args.max_new_tokens,
                        temp=args.temperature,
                        top_k=args.top_k)
        outf.write(f"{q}\t{sql}\n")
        outf.flush()

print(f"\n✅ Predictions written to {out_path}")
