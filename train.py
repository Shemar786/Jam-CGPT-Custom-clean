"""
This training script can run on a single GPU or distributed (DDP).

Single-GPU example:
    python train.py --batch_size=32 --compile=False

DDP on 4 GPUs (1 node):
    torchrun --standalone --nproc_per_node=4 train.py

DDP across 2 nodes (master 123.456.123.456):
    # Master node
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
    # Worker node
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

If your cluster lacks Infiniband, prepend NCCL_IB_DISABLE=1.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, get_lora_model

# -----------------------------------------------------------------------------
# Default config values (overridden by config file / CLI)
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 40
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'            # 'scratch', 'resume', 'gpt2*', or '/path/to/ckpt.pt'
# wandb
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5
batch_size = 12
block_size = 1024
# model size
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# LoRA
lora_rank = 0
lora_alpha = 0.0
lora_dropout = 0.0
compute_grad_memory = False
# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP
backend = 'nccl'
# system
device = 'cpu'
dtype = 'bfloat16'
compile = True
freeze = False
outfilename = 'ckpt_pretrain.pt'
# -----------------------------------------------------------------------------
# Override from config/CLI
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int,float,bool,str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP setup ------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8  # simulate 8 GPUs
print("total number of tokens per iteration:", batch_size * block_size * gradient_accumulation_steps)
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Reproducibility & precision -----------------------------------
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading ---------------------------------------------------
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# counters -------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# infer vocab size ----------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path,'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ----------------------------------------------------------------------------
# Build or load model BEFORE moving to device
# ----------------------------------------------------------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
checkpoint_iter_num = 0

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size or 50304
    model = GPT(GPTConfig(**model_args))

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, outfilename)
    checkpoint = torch.load(ckpt_path, map_location=device)
    cma = checkpoint['model_args']
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size','lora_rank','lora_alpha']:
        model_args[k] = cma.get(k, model_args.get(k))
    model = GPT(GPTConfig(**model_args))
    sd = checkpoint['model']
    pref = '_orig_mod.'
    for key in list(sd.keys()):
        if key.startswith(pref):
            sd[key[len(pref):]] = sd.pop(key)
    model.load_state_dict(sd)
    iter_num = checkpoint['iter_num']
    checkpoint_iter_num = iter_num

elif init_from.startswith('gpt2'):
    print(f"Initializing from GPT-2: {init_from}")
    override_args = dict(dropout=dropout, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size','lora_rank','lora_alpha']:
        model_args[k] = getattr(model.config, k)

elif os.path.isfile(init_from):
    print(f"Loading pretrained checkpoint from {init_from}")
    checkpoint = torch.load(init_from, map_location=device)
    cma = checkpoint['model_args']
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size','lora_rank','lora_alpha']:
        model_args[k] = cma.get(k, model_args.get(k))
    model = GPT(GPTConfig(**model_args))
    sd = checkpoint['model']
    pref = '_orig_mod.'
    for key in list(sd.keys()):
        if key.startswith(pref):
            sd[key[len(pref):]] = sd.pop(key)
    model.load_state_dict(sd)
    print("Checkpoint loaded.")

else:
    raise ValueError(f"init_from '{init_from}' not recognized.")

model.to(device)

# scaler & optimizer -------------------------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1,beta2), device_type)
if init_from=='resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile & DDP wrap -------------------------------------------
if compile:
    print("compiling the model... (takes ~1 min)")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# loss estimation ----------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X,Y = get_batch(split)
            with ctx:
                _, loss = model(X,Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# lr schedule --------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# wandb --------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop ------------------------------------------------
X,Y = get_batch('train')
t0 = time.time()
local_iter = 0
raw = model.module if ddp else model
running_mfu = -1.0

while True:
    # set lr
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # eval & save
    if (iter_num % eval_interval == 0 and master_process) or (iter_num == checkpoint_iter_num and master_process):
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                'iter': iter_num,
                'train/loss': losses['train'],
                'val/loss': losses['val'],
                'lr': lr,
                'mfu': running_mfu * 100,
            })
        if (losses['val'] < best_val_loss) or always_save_checkpoint or (iter_num == checkpoint_iter_num):
            best_val_loss = losses['val']
            ckpt_data = {
                'model': raw.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(ckpt_data, os.path.join(out_dir, outfilename))

    # forward/backward
    for micro in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing & logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if local_iter >= 5:
        mfu = raw.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = running_mfu if running_mfu != -1.0 else mfu
    if local_iter % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
