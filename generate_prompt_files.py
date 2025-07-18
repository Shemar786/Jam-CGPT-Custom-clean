import os
import csv

# Paths
csv_file = 'datasets/spider-text-sql/spider_text_sql.csv'  # Update this if needed
output_dir = 'data/spider_textsql_samples'

# Create the output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read and write prompts
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        if len(row) < 2:
            continue  # skip malformed rows
        eng, sql = row[0].strip(), row[1].strip()
        prompt = f"ENG:\t{eng}\nSQL:\t{sql}"
        filename = os.path.join(output_dir, f"{idx:05d}.txt")
        with open(filename, 'w', encoding='utf-8') as out:
            out.write(prompt)

print(f"✅ Done! Wrote {idx + 1} prompt files to {output_dir}")
