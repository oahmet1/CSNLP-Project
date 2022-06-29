
## THIS FILE IS JUST TO BE MERGED WITH THE ONE THAT IS ON DAVID'S COMPUTER!!!



## MNLI
### lsf.o223751116
output_dir_dm_static_mnli
rm -rf ../../output_dir_dm_static_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_dm_static_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 0


### lsf.o223751121
output_dir_dm_static_mnli2


### lsf.o223751133
output_dir_dm_fixed_mnli
rm -rf ../../output_dir_dm_fixed_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_dm_fixed_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 0


### lsf.o223751137
output_dir_dm_fixed_mnli2







### lsf.223751573
output_dir_amr0_mnli
rm -rf ../../output_dir_amr0_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_amr0_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 0

### lsf.223751607
output_dir_amr0_mnli2



### lsf.223751656
output_dir_amr0_static_mnli
rm -rf ../../output_dir_amr0_static_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_amr0a_static_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 0


### lsf.223751664
output_dir_amr0_static_mnli2



### lsf.223751754
output_dir_amr0_fixed_mnli
rm -rf ../../output_dir_amr0_fixed_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_amr0_fixed_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 0

### lsf.223751775
output_dir_amr0_fixed_mnli2








### lsf.223751789
output_dir_amr1_mnli
rm -rf ../../output_dir_amr1_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_amr1_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 1

### lsf.223751795
output_dir_amr1_mnli2



### lsf.223751801
output_dir_amr1_static_mnli
rm -rf ../../output_dir_amr1_static_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_amr1_static_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 1


### lsf.223751811
output_dir_amr1_static_mnli2



### lsf.223751835
output_dir_amr1_fixed_mnli
rm -rf ../../output_dir_amr1_fixed_mnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task mnli --data_dir ../../data/glue_data/MNLI --output_dir ../../output_dir_amr1_fixed_mnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 0 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 1


### lsf.223751848
output_dir_amr1_fixed_mnli2



---------------------------------------------------------------------

## QNLI
### lsf.o223750671
output_dir_amr0_qnli
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
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 0

### lsf.o223750789
output_dir_amr0_qnli2

### lsf.o223750794
output_dir_amr0_qnli3



### lsf.o223750849
output_dir_amr0_static_qnli
rm -rf ../../output_dir_amr0_static_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_amr0_static_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 3 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 0


### lsf.o223750858
output_dir_amr0_static_qnli2



### lsf.o223750880
output_dir_amr0_fixed_qnli
rm -rf ../../output_dir_amr0_fixed_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_amr0_fixed_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 3 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 0

### lsf.o223750883
output_dir_amr0_fixed_qnli2



### lsf.o223750950
output_dir_amr1_qnli
rm -rf ../../output_dir_amr1_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_amr1_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 3 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 1


### lsf.o223750954
output_dir_amr1_qnli2

### lsf.o223750956
output_dir_amr1_qnli3



### lsf.o223750965
output_dir_amr1_static_qnli
rm -rf ../../output_dir_amr1_static_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_amr1_static_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 3 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 1

### lsf.o223750969
output_dir_amr1_static_qnli2



### lsf.o223750985
output_dir_amr1_fixed_qnli
rm -rf ../../output_dir_amr1_fixed_qnli

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task qnli --data_dir ../../data/glue_data/QNLI --output_dir ../../output_dir_amr1_fixed_qnli  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 3 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 1


### lsf.o223751009
output_dir_amr1_fixed_qnli2


-------------------------------------------------------------------------------


## RTE
### lsf.o223750214
output_dir_amr0_rte
rm -rf ../../output_dir_amr0_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_amr0_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 0

### lsf.o223750246
output_dir_amr0_rte2

### lsf.o223750253
output_dir_amr0_rte3



### lsf.o223750295
output_dir_amr0_static_rte
rm -rf ../../output_dir_amr0_static_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_amr0_static_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 0


### lsf.o223750310
output_dir_amr0_static_rte2

### lsf.o223750331
output_dir_amr0_static_rte3



### lsf.o223750347
output_dir_amr0_fixed_rte
rm -rf ../../output_dir_amr0_fixed_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_amr0_fixed_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 0

### lsf.o223750352
output_dir_amr0_fixed_rte2

### lsf.o223750356
output_dir_amr0_fixed_rte3



### lsf.o223750390
output_dir_amr1_rte
rm -rf ../../output_dir_amr1_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_amr1_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 0 \
    --amr_version 1

### lsf.o223750427
output_dir_amr1_rte2

### lsf.o223750450
output_dir_amr1_rte3



### lsf.o223750457
output_dir_amr1_static_rte
rm -rf ../../output_dir_amr1_static_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_amr1_static_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 1 --fixed_encoder 0 \
    --amr_version 1

### lsf.o223750525
output_dir_amr1_static_rte2

### lsf.o223750546
output_dir_amr1_static_rte3



### lsf.o223750585
output_dir_amr1_fixed_rte
rm -rf ../../output_dir_amr1_fixed_rte

TOKENIZERS_PARALLELISM=true
module load gcc/8.2.0 python_gpu/3.9.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train.py \
    --do_train --task rte --data_dir ../../data/glue_data/RTE --output_dir ../../output_dir_amr1_fixed_rte  \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism amr --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 1 --numworkers 4 --static_embeddings 0 --fixed_encoder 1 \
    --amr_version 1

### lsf.o223750590
output_dir_amr1_fixed_rte2

### lsf.o223750595
output_dir_amr1_fixed_rte3


