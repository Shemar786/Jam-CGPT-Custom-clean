# ── START SCRIPT ───────────────────────────────────────────────
#!/usr/bin/env python3
"""
Prepare a folder of prompt .txt files for Jam-CGPT fine-tuning.
Each .txt file should look like:

ENG:   <natural-language question>
SQL:   <corresponding SQL query>

The script
  1) splits files into train / val
  2) tokenises with TikToken GPT-2 encoder
  3) writes train.bin and val.bin
"""

import os, argparse, numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True,
                   help="folder containing *.txt prompt files")
    p.add_argument("--output-dir", default="bins/",
                   help="where train.bin and val.bin are written")
    p.add_argument("--val-ratio",  type=float, default=0.10)
    p.add_argument("--num-proc",   type=int,   default=4)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted([
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.endswith(".txt")
    ])
    if not all_files:
        raise ValueError(f"No .txt files in {args.input_dir}")

    val_n   = int(len(all_files) * args.val_ratio)
    files   = {"val": all_files[:val_n], "train": all_files[val_n:]}
    ds      = load_dataset("text", data_files=files, sample_by="document")

    enc = tiktoken.get_encoding("gpt2")
    def tok(ex):
        ids = enc.encode_ordinary(ex["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    ds_tok = ds.map(tok, remove_columns=["text"],
                    num_proc=args.num_proc, desc="Tokenising")

    dtype = np.uint16
    for split, d in ds_tok.items():
        total = int(np.sum(d["len"]))
        out   = os.path.join(args.output_dir, f"{split}.bin")
        arr   = np.memmap(out, dtype=dtype, mode="w+", shape=(total,))
        print(f"Writing {out}  ({total} tokens)…")
        idx = 0
        for e in tqdm(d, total=len(d)):
            arr[idx : idx + e["len"]] = e["ids"]
            idx += e["len"]
        arr.flush()
    print("✅  Done – binaries are in", args.output_dir)

if __name__ == "__main__":
    main()
# ── END SCRIPT ─────────────────────────────────────────────────
