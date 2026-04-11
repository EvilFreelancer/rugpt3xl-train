#!/bin/bash
cd /home/pasha/train/docker-unsloth/work

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export NCCL_TIMEOUT=3600
export FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "Starting training at $(date)" >> training_fsdp.log
torchrun --nproc_per_node=4 train_rugpt3xl_fsdp.py 2>&1 | tee -a training_fsdp.log
echo "Training finished at $(date)" >> training_fsdp.log
