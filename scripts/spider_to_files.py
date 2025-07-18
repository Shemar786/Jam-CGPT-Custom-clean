"""
Convert the Spider CSV (text_query, sql_command) into 10 000 paired .txt files
so Jam-CGPT’s prepare script can treat them like TDAT/COM examples.

Creates:
  data/spider_files/10k_nl/   – English question per file
  data/spider_files/10k_sql/  – matching SQL query per file
"""

import csv, argparse, pathlib, tqdm

def main(csv_path: str, out: str, limit: int):
    sql_dir = pathlib.Path(out, "10k_sql")
    nl_dir  = pathlib.Path(out, "10k_nl")
    sql_dir.mkdir(parents=True, exist_ok=True)
    nl_dir.mkdir(parents=True,  exist_ok=True)

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(tqdm.tqdm(reader, desc="writing")):
            if i >= limit:
                break
            nl  = row["text_query"].strip()      # column in CSV
            sql = row["sql_command"].strip()     # column in CSV
            (nl_dir  / f"{i:05d}.txt").write_text(nl  + "\n")
            (sql_dir / f"{i:05d}.txt").write_text(sql + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  dest="csv_path",
                   default="data/spider_raw/spider_text_sql.csv",
                   help="Path to spider_text_sql.csv")
    p.add_argument("--out",  default="data/spider_files",
                   help="Output folder for 10k_nl/ and 10k_sql/")
    p.add_argument("--limit", type=int, default=10000,
                   help="How many examples to extract")
    main(**vars(p.parse_args()))

