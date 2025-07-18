# ── Fine-tune SO13M model on English→SQL prompts ──
import time

# ---------- where to save ----------
out_dir       = 'out-so13m-finetuned'                 # new fine-tuned output folder
outfilename   = 'ckpt_sql8k_so13m_finetuned.pt'       # new checkpoint filename
init_from     = 'pretrained_so13m/ckpt.pt'            # pretrained SO13M checkpoint

# ---------- data ----------
dataset   = 'sql8k'
train_bin = '../bins/train.bin'    # path relative to *this* file
val_bin   = '../bins/val.bin'

# ---------- tiny model ----------
n_layer = 6
n_head  = 8
n_embd  = 512
block_size = 256

# ---------- training ----------
batch_size  = 10
gradient_accumulation_steps = 16
max_iters   = 25_000
learning_rate = 5e-4
decay_lr    = True
warmup_iters = 200

# ---------- logging ----------
eval_interval = 200
eval_iters    = 40
wandb_log     = True
wandb_project = 'jam-cgpt'
wandb_run_name = f'sql8k_so13m_finetuned_{int(time.time())}'

# ---------- precision ----------
dtype = 'float16'
