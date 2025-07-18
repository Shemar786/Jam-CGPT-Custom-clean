#!/usr/bin/env bash
set -euo pipefail

QFILE=test_questions.txt
SO=so13m_only.tsv
SC=scratch_only.tsv
OUT_TSV=side_by_side.tsv
OUT_CSV=side_by_side.csv

extract_sql() {
  awk '
    /^SQL:/     {sub(/^SQL:[[:space:]]*/,""); print; found=1; exit}
    /^[Ss][Ee][Ll][Ee][Cc][Tt]/ {print; found=1; exit}
    /^[Ww][Ii][Tt][Hh]/ {print; found=1; exit}
    {last=$0}
    END{if(!found)print last}
  '
}

# --- SO13M model ---
: > "$SO"
while IFS= read -r q; do
  [[ -z "$q" ]] && continue
  python3 sample_so13m_finetuned.py --prompt "$q" \
    | extract_sql \
    | awk -vQ="$q" '{printf "%s\t%s\n",Q,$0}' >> "$SO"
done < "$QFILE"

# --- Scratch model ---
: > "$SC"
while IFS= read -r q; do
  [[ -z "$q" ]] && continue
  python3 sample_jam_cgpt.py \
    config/finetune_small_sql8k.py \
    --out_dir out-sql8k-scratch \
    --outfilename ckpt_sql8k_scratch.pt \
    --prompt "$q" \
    | extract_sql \
    | awk -vQ="$q" '{printf "%s\t%s\n",Q,$0}' >> "$SC"
done < "$QFILE"

# Combine into 3-col TSV
paste -d$'\t' "$SO" <(cut -f2 "$SC") > "$OUT_TSV"

# Safe quoted CSV for Sheets/Excel
python3 - <<'PY'
import csv
rows=[["Question","SO13M_SQL","Scratch_SQL"]]
with open("side_by_side.tsv",encoding="utf-8") as f:
    for line in f:
        parts=line.rstrip("\n").split("\t")
        if len(parts)<3: parts+=[""]*(3-len(parts))
        rows.append(parts[:3])
with open("side_by_side.csv","w",newline="",encoding="utf-8") as out:
    csv.writer(out,quoting=csv.QUOTE_ALL).writerows(rows)
PY

echo "Done. Wrote side_by_side.tsv and side_by_side.csv"
