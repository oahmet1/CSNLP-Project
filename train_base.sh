#!/usr/bin/zsh
rm -r output_dir

#TOKENIZERS_PARALLELISM=true

python train.py \
    --do_train --task qnli --data_dir data/glue_data/QNLI --output_dir output_dir \
    --model_name_or_path roberta-base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 8 --gradient_accumulation_steps 4 --eval_batch_size 16 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 10
