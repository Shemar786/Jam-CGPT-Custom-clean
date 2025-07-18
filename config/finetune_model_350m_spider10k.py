# ──────────────────── Spider-10k fine-tune on GPT-2 350M ────────────────────
import time

# ─────────────────── Checkpoint locations ───────────────────
out_dir       = 'out-jam-cgpt'        # dir that holds ckpt_pretrain.pt
outfilename   = 'ckpt_retrain.pt'     # will be overwritten on improvements
init_from     = 'resume'            # start from ckpt_pretrain.pt

# ─────────────────── Logging / evaluation ───────────────────
eval_interval = 100                   # how often to eval
eval_iters    = 80                    # batches to average val loss
wandb_log     = True
wandb_project = 'jam-cgpt'
wandb_run_name = 'jam-cgpt-model350m-spider10k'

# ──────────────────── Dataset paths ─────────────────────────
dataset   = 'spider10k'                          # just a label for logs
train_bin = 'bins/spider10k/train.bin'           # built via spider_prepare.py
val_bin   = 'bins/spider10k/val.bin'

# ──────────────────── Save-every-improvement flag ───────────
always_save_checkpoint = True

# ─────────────────── Model / tokenizer ──────────────────────
block_size = 256                                  # max sequence length

# ─────────────────── Training hyper-parameters ─────────────
batch_size = 4
gradient_accumulation_steps = 32                  # effective batch = 128
max_iters = 272000 + 1150 * 3                     # (authors’ table)
learning_rate = 3e-5
decay_lr = False                                  # constant LR

# ─────────────────── Mixed precision ───────────────────────
dtype = 'float16'                                 # AMP precision

