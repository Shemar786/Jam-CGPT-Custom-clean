import os
import json
import pickle
import tiktoken

def load_jsonl(path: str):
    """
    Read a JSON-lines file into a list of dicts.
    Each line in the file should be a separate JSON object.
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def process(split: str):
    """
    Convert the Spider JSON-lines split (train or dev) into a binary
    file of token IDs for fine-tuning.
    """
    # Load the JSON-lines data
    data = load_jsonl(f"data/spider/{split}.json")

    # Initialize GPT-2 BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Fetch the ID for the end-of-text special token, allowing it explicitly
    special_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    all_ids = []
    for item in data:
        # Assume item is a list: [question, sql]
        if not isinstance(item, list) or len(item) < 2:
            print(f"Skipping invalid item: {item}")
            continue
        question = str(item[0]).strip()  # First element is the question
        sql = str(item[1]).strip()       # Second element is the SQL query

        # Tokenize question and SQL separately
        ids_q = enc.encode(question)
        ids_sql = enc.encode(sql)

        # Concatenate: question tokens + end-of-text token + SQL tokens
        all_ids.extend(ids_q + [special_id] + ids_sql)

    # Write out the token list as a pickled binary
    os.makedirs("bins", exist_ok=True)
    out_path = f"bins/{split}.bin"
    with open(out_path, "wb") as out:
        pickle.dump(all_ids, out)

    print(f"✅ bins/{split}.bin: {len(all_ids)} tokens")

if __name__ == "__main__":
    # Process both splits
    process("train")  # writes bins/train.bin
    process("dev")    # writes bins/dev.bin
