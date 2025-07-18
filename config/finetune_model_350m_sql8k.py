k# ── Fine-tune a tiny GPT from scratch on 8 k text-to-SQL samples ──
import time

# ---------- where to save ----------
out_dir       = 'out-sql8k-scratch'
outfilename   = 'ckpt_sql8k_scratch.pt'
init_from     = 'scratch'          # IMPORTANT

# ---------- data ----------
dataset   = 'sql8k'
train_bin = '../bins/train.bin'    # adjust if path differs
val_bin   = '../bins/val.bin'

# ---------- tiny model ----------
n_layer = 6
n_head  = 8
n_embd  = 512          # model size ≈ 52 M params
block_size = 256

# ---------- training ----------
batch_size  = 8         # fits easily on 8 GB VRAM
gradient_accumulation_steps = 16
max_iters   = 25_000    # you can stop early if val loss plateaus
learning_rate = 5e-4    # a bit higher since we start from scratch
decay_lr    = True
warmup_iters = 200

# ---------- logging ----------
eval_interval = 200
eval_iters    = 40
wandb_log     = True
wandb_project = 'jam-cgpt'
wandb_run_name = f'sql8k_scratch_{int(time.time())}'

dtype = 'float16'       # switch to 'float32' if fp16 underflows

