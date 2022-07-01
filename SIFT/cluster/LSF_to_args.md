
# Euler LSF execution map

## In this Document we keep a mapping of lsf numbers to execution settings and arguments for reproducibility purposes


example:

### lsf.o221967629:
    - bla
    - bla

---

### lsf.o222521532   (QNLI baseline with default settings and roberta-base)

rm -rf ../../output_dir_dm_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_dm_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base  --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4



### lsf.o222906246
same as above
output_dir_dm_qnli2


### lsf.o222906380
same as above
output_dir_dm_qnli3


### lsf.o222962794 (QNLI baseline run1 with static embeddings)
rm -rf ../../output_dir_dm_static_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_dm_static_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 1 --fixed_encoder 0

### lsf.o222963188 (QNLI baseline run2 with static embeddings)
rm -rf ../../output_dir_dm_static_qnli


### lsf.o222964981 (QNLI baseline run1 with fixed encoder)
rm -rf ../../output_dir_dm_fixed_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_dm_fixed_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 1

### lsf.222964678 (QNLI baseline run2 with fixed encoder)
rm -rf ../../output_dir_dm_fixed_qnli2





### lsf.o222519606   (MNLI baseline with default settings and roberta-large)
rm -rf ../../output_dir_dm_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_dm_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_large  --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4

-> CRASHED


### lsf.o222521580   (MNLI baseline with default settings and roberta-base)
rm -rf ../../output_dir_dm_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_dm_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base  --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4

-> CRASHED

### lsf.o222864213
output_dir_dm_mnli4
same as above but with numworkers 1

### lsf.o222869792
output_dir_dm_mnli5
same as above but with numworkers 0



### lsf.o222521555   (RTE baseline with default settings and roberta-base)
rm -rf ../../output_dir_dm_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_dm_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4


### lsf.o222907002 (RTE baseline run2)
output_dir_dm_rte2


### lsf.o222907186 (RTE baseline run3)
output_dir_dm_rte3

### lsf.o222915034 (RTE baseline run1 with static embeddings)
rm -rf ../../output_dir_dm_static_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_dm_static_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 1 --fixed_encoder 0


### lsf.o222915438 (RTE baseline run2 with static embeddings)
output_dir_dm_static_rte2

### lsf.o222915590 (RTE baseline run3 with static embeddings)
output_dir_dm_static_rte3


### lsf.o222916249 (RTE baseline run1 with fixed encoder)
rm -rf ../../output_dir_dm_fixed_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_dm_fixed_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 1

### lsf.o222916599 (RTE baseline run2 with fixed encoder)
../../output_dir_dm_fixed_rte2

### lsf.o222916746 (RTE baseline run3 with fixed encoder)
../../output_dir_dm_fixed_rte3





