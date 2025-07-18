#!/usr/bin/env python3
"""
sample_so13m_finetuned.py
=========================

• SINGLE‑PROMPT MODE
    python3 sample_so13m_finetuned.py --prompt "Show the status …"
  → prints ONLY the SQL string.

• BATCH MODE (omit --prompt)
  Samples a handful of rows from the Spider CSV and writes
  jam_cgpt_predictions/my_samples_so13m.txt
"""

import argparse, csv, pathlib, torch, tiktoken
from types import SimpleNamespace
from model import GPT

# ── helper: build model ────────────────────────────────────────
def load_model(ckpt_path: str, device: str = "cpu") -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = SimpleNamespace(**ckpt["model_args"])
    cfg.lora_dropout = 0.0
    cfg.attn_pdrop   = getattr(cfg, "attn_pdrop", cfg.dropout)
    cfg.resid_pdrop  = getattr(cfg, "resid_pdrop", cfg.dropout)

    clean_state = {k.replace("_orig_mod.", ""): v
                   for k, v in ckpt["model"].items()}

    model = GPT(cfg)
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

# ── helper: encode / decode ───────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: torch.tensor([enc.encode(s)], dtype=torch.long)
decode = lambda ids: enc.decode(ids)

def generate(model: GPT,
             prompt: str,
             max_new_tokens: int = 64,
             temperature: float = 0.2,
             device: str = "cpu") -> str:
    x = encode(prompt).to(device)
    with torch.no_grad():
        y = model.generate(x,
                           max_new_tokens=max_new_tokens,
                           temperature=temperature)
    return decode(y[0].tolist())

# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="ONE English question → SQL")
    args   = parser.parse_args()

    # paths / defaults
    ckpt_path      = "out-so13m-finetuned/ckpt_sql8k_so13m_finetuned.pt"
    in_csv         = "datasets/spider-text-sql/spider_text_sql.csv"
    out_file       = "jam_cgpt_predictions/my_samples_so13m.txt"
    num_samples    = 5
    max_new_tokens = 64
    temperature    = 0.2
    device         = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(ckpt_path, device).to(device)

    # ── SINGLE‑PROMPT MODE ────────────────────────────────────
    if args.prompt:
        sql = generate(model, args.prompt.strip(),
                       max_new_tokens, temperature, device).strip()

        # strip echoed "SQL:" prefix and <|endoftext|> tail if present
        if "SQL:" in sql:
            sql = sql.split("SQL:", 1)[-1].strip()
        if "<|endoftext|>" in sql:
            sql = sql.split("<|endoftext|>", 1)[0].strip()

        print(sql)
        raise SystemExit(0)

    # ── BATCH MODE (for quick samples) ───────────────────────
    rows = list(csv.reader(open(in_csv)))[1:num_samples+1]  # skip header
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        for q, _ in rows:
            pred_sql = generate(model, q,
                                max_new_tokens, temperature, device).strip()
            f.write(f"{q}\t{pred_sql}\n")
            print(f"- {q}\n  → {pred_sql}\n")

    print(f"\n✅ Predictions written to {out_file}")
