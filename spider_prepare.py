#!/usr/bin/env python3
"""
Convert Kaggle “spider_text_sql.csv” into GPT-2 token-ID .bin files
usable by Jam-CGPT (ENG → SQL).

• 90 % of rows → train.bin
• 10 % of rows → val.bin
• Tokenizer  : GPT-2 (tiktoken)
• Output dir : whatever you pass via --outdir (e.g. bins/spider10k)

Run:
  python3 spider_prepare.py \
          --csv    datasets/spider-text-sql/spider_text_sql.csv \
          --outdir bins/spider10k
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import tiktoken

# ----- column names in the Kaggle CSV ----------------------------------------
QUESTION_COL = "text_query"   # English question
SQL_COL      = "sql_command"  # SQL query
# -----------------------------------------------------------------------------

ENC      = tiktoken.get_encoding("gpt2")
END_TOKEN = "<|endoftext|>"

def encode(text: str):
    """Encode text with GPT-2 tokenizer, allowing the special end token."""
    return ENC.encode(text, allowed_special={END_TOKEN})

def make_bin(csv_path: str, out_dir: str,
             train_frac: float = 0.9, seed: int = 42):
    """Read CSV → build train.bin / val.bin."""
    df = pd.read_csv(csv_path)
    assert {QUESTION_COL, SQL_COL} <= set(df.columns), (
        f"CSV must contain columns '{QUESTION_COL}' and '{SQL_COL}'"
    )

    # deterministic shuffle
    rng = random.Random(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    df = df.loc[idx].reset_index(drop=True)

    split_idx = int(len(df) * train_frac)
    splits = {"train": df.iloc[:split_idx], "val": df.iloc[split_idx:]}

    os.makedirs(out_dir, exist_ok=True)

    for split_name, subset in splits.items():
        toks = []
        for _, row in subset.iterrows():
            example = (
                f"ENG:\t{row[QUESTION_COL].strip()}\n"
                f"SQL:\t{row[SQL_COL].strip()}{END_TOKEN}"
            )
            toks.extend(encode(example))

        out_path = f"{out_dir}/{split_name}.bin"
        np.array(toks, dtype=np.uint16).tofile(out_path)
        print(f"Wrote {split_name}.bin  ({len(toks):,} tokens)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    required=True, help="Path to spider_text_sql.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for .bin files")
    ap.add_argument("--train_frac", type=float, default=0.9,
                    help="Fraction of rows for train split (default 0.9)")
    ap.add_argument("--seed",  type=int, default=42, help="Shuffle seed (default 42)")
    args = ap.parse_args()

    make_bin(args.csv, args.outdir, args.train_frac, args.seed)
