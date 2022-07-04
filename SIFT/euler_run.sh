#!/usr/bin/bash

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPs_PROXY=http://proxy.ethz.ch:3128

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# use better solution when running multiple jobs!
# rm -rf ../../output_dir


rm -rf ../../output_dir_amr0_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_amr0_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 0
