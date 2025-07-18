#!/usr/bin/env python3
"""
Prepare a folder of prompt .txt files for Jam-CGPT fine-tuning.

Each .txt file should look like:

ENG:   <natural-language question>
SQL:   <corresponding SQL query>

The script:
1. Splits the files into train/val (deterministic).
2. Loads them with Hugging Face `datasets` (sample_by='document').
3. Tokenises with TikToken GPT-2 encoder, appending the EOT token.
4. Writes `train.bin` and `val.bin` (uint16) to --output-dir.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import tiktoken

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert prompt .txt files into train.bin / val.bin for Jam-CGPT"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing *.txt prompt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bins/",
        help="Where to write train.bin and val.bin",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of files used for validation (default 0.10)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of CPU processes for tokenisation",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Collect files and make deterministic split
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted(
        [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".txt")
        ]
    )
    if not all_files:
        raise ValueError(f"No .txt files found in {args.input_dir}")

    val_sz = int(len(all_files) * args.val_ratio)
    val_files = all_files[:val_sz]
    train_files = all_files[val_sz:]

    data_files = {"train": train_files, "val": val_files}
    dataset = load_dataset("text", data_files=data_files, sample_by="document")

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")

    def _process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)          # end-of-text token
        return {"ids": ids, "len": len(ids)}

    tokenised = dataset.map(
        _process,
        remove_columns=["text"],
        desc="Tokenising",
        num_proc=args.num_proc,
    )

    # ------------------------------------------------------------------
    # 3. Write .bin files
    # ------------------------------------------------------------------
    dtype = np.uint16  # GPT-2 BPE ids fit in uint16
    for split, dset in tokenised.items():
        total_len = int(np.sum(dset["len"]))
        out_path = os.path.join(args.output_dir, f"{split}.bin")
        arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total_len,))

        print(f"Writing {out_path}  ({total_len} tokens)…")
        idx = 0
        for example in tqdm(dset, total=len(dset)):
            arr[idx : idx + example["len"]] = example["ids"]
            idx += example["len"]
        arr.flush()

    print("✅  Done.  Binaries are in", args.output_dir)

if __name__ == "__main__":
    main()

