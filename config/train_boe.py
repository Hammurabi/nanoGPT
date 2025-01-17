# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-10M'
dataset = 'shakespeare'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 2
block_size = 768
gradient_accumulation_steps = 1

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100
eval_iters = 20
log_interval = 10

# weight decay
weight_decay = 1e-1
    
n_layer: int = 1
n_head: int = 1
n_embd: int = 96
dropout: float = 0.1
use_moe = True
num_experts = 384*2
num_experts_per_tok = 6
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = False # use PyTorch 2.0 to compile the model to be faster
always_save_checkpoint = False # if True, always save a checkpoint after each eval
# dtype = 'bfloat16' # 'float32', 'float16', 'bfloat16'
# ptdtype = 'bfloat16' # 'float32', 'float16', 'bfloat16'

import torch
print(torch.__version__)