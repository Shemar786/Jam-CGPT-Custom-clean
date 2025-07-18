#!/usr/bin/env python3
"""
compare_sql_models.py
Runs the SAME English questions through two models and writes a CSV:
  English question | SO13‑M SQL | Scratch SQL
"""

import subprocess, csv, pathlib

TEST_FILE = "test_questions.txt"
OUT_CSV   = "side_by_side.csv"

# -------- helper to call a script --------
def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()

# -------- start fresh --------
pathlib.Path(OUT_CSV).unlink(missing_ok=True)

questions = [q.strip() for q in open(TEST_FILE) if q.strip()]
for q in questions:
    so13_sql = run([
        "python3", "sample_so13m_finetuned.py",
        "--prompt", q
    ])

    scratch_sql = run([
        "python3", "sample_jam_cgpt.py",
        "config/finetune_small_sql8k.py",
        "--out_dir", "out-sql8k-scratch",
        "--outfilename", "ckpt_sql8k_scratch.pt",
        "--prompt", q
    ])

    with open(OUT_CSV, "a", newline="") as f:
        csv.writer(f).writerow([q, so13_sql, scratch_sql])

print(f"✅ Comparison written to {OUT_CSV}")
