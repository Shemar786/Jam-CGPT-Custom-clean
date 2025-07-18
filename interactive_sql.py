#!/usr/bin/env python3
import torch
from model import GPT, GPTConfig
import tiktoken
import argparse

# ── parse args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interactive English→SQL prompt for Jam-CGPT")
parser.add_argument("config_file", help="Path to the model config file (defines n_layer, n_head, etc.)")
parser.add_argument("--checkpoint",
                    default="out-jam-cgpt/ckpt_pretrain.pt",
                    help="Path to the trained checkpoint")
parser.add_argument("--max_new_tokens", type=int, default=64,
                    help="Maximum tokens to generate in response")
args = parser.parse_args()

# ── load user config file by exec’ing it into a dict ────────────
cfg = {}
try:
    with open(args.config_file, 'r') as f:
        exec(f.read(), cfg)
except FileNotFoundError:
    print(f"Error: Config file {args.config_file} not found.")
    exit(1)

# ── build the GPTConfig from variables in cfg (or fall back) ────
# your finetune_small_sql8k.py defines: n_layer, n_head, n_embd, block_size, dropout
config = GPTConfig(
    n_layer   = cfg.get('n_layer', 24),
    n_head    = cfg.get('n_head', 16),
    n_embd    = cfg.get('n_embd', 1024),
    vocab_size= cfg.get('vocab_size', 50257),   # GPT-2 vocab size
    block_size= cfg.get('block_size', 256),
    dropout   = cfg.get('dropout', 0.1),
)

# ── device setup ────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ── instantiate & load model ───────────────────────────────────
model = GPT(config)
try:
    ckpt = torch.load(args.checkpoint, map_location=device)
    # checkpoint may store under 'model' key or be the raw state_dict
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=False)
except FileNotFoundError:
    print(f"Error: Checkpoint file {args.checkpoint} not found.")
    exit(1)
model.to(device)
model.eval()

# ── parameter count ─────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params/1e6:.2f}M")

# ── tokenizer setup ────────────────────────────────────────────
tok        = tiktoken.get_encoding("gpt2")
eot_token  = tok.eot_token
print(f"Tokenizer: gpt2, vocab size: {tok.n_vocab}, eot_token: {eot_token}")

print(f"\nInteractive SQL generator (max_new_tokens={args.max_new_tokens})")
print("Type your English question. Press Ctrl-D or Ctrl-C to quit.\n")

# ── interactive loop ───────────────────────────────────────────
while True:
    try:
        q = input("ENG: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break
    if not q:
        continue

    # build the prompt exactly as your finetune_data used
    prompt = f"ENG:\t{q}\nSQL:\t"
    print(f"Prompt: {prompt!r}")

    # tokenize
    ids = tok.encode(prompt)
    X   = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"Input tokens (first 10): {ids[:10]}")

    # generate
    with torch.no_grad():
        out = model.generate(X, max_new_tokens=args.max_new_tokens)
    full_ids = out[0].tolist()
    gen_ids  = full_ids[len(ids):]
    print(f"Generated tokens after prompt (first 10): {gen_ids[:10]}")

    # stop at EOT if present
    if eot_token in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eot_token)]
        print(f"Stopped at eot_token: {eot_token}")

    # decode
    generated = tok.decode(gen_ids)
    print(f"Decoded output: {generated!r}")

    # strip whitespace and show final SQL
    print("→", generated.strip(), "\n")

