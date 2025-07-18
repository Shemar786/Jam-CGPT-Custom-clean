#!/usr/bin/env bash
set -euo pipefail

QFILE=test_questions.txt
SO=so13m_only.tsv
SC=scratch_only.tsv

extract_sql() {
  awk '
    /^SQL:/     {sub(/^SQL:[[:space:]]*/,""); print; exit}
    /^[Ss][Ee][Ll][Ee][Cc][Tt]/ {print; exit}
    /^[Ww][Ii][Tt][Hh]/ {print; exit}
    {last=$0}
    END{print last}
  '
}

# SO13M
: > "$SO"
echo "Generating SO13M outputs..."
count=0
while read -r q; do
  ((++count))
  echo " SO13M  [$count] $q"
  sql=$(python3 sample_so13m_finetuned.py --prompt "$q" | extract_sql)
  printf "%s\t%s\n" "$q" "$sql" >> "$SO"
done < "$QFILE"

# Scratch
: > "$SC"
echo "Generating Scratch outputs..."
count=0
while read -r q; do
  ((++count))
  echo " Scratch [$count] $q"
  sql=$(python3 sample_jam_cgpt.py \
        config/finetune_small_sql8k.py \
        --out_dir out-sql8k-scratch \
        --outfilename ckpt_sql8k_scratch.pt \
        --prompt "$q" | extract_sql)
  printf "%s\t%s\n" "$q" "$sql" >> "$SC"
done < "$QFILE"

# Combine & quote
paste -d$'\t' "$SO" <(cut -f2 "$SC") > side_by_side.tsv
python3 - <<'PY'
import csv
rows=[["Question","SO13M_SQL","Scratch_SQL"]]
with open("side_by_side.tsv", encoding="utf-8") as f:
    for line in f:
        parts=line.rstrip("\n").split("\t")
        rows.append(parts + [""]*(3-len(parts)))
with open("side_by_side.csv","w",newline="",encoding="utf-8") as out:
    csv.writer(out, quoting=csv.QUOTE_ALL).writerows(rows)
print("Done: side_by_side.tsv + side_by_side.csv")
PY
