# Running on ETH's Euler Cluster

## Link Collection

#### "summary"
- https://scicomp.ethz.ch/wiki/LSF_mini_reference

####  rest
- https://scicomp.ethz.ch/wiki/Python
- https://scicomp.ethz.ch/wiki/Getting_started_with_clusters



## How to run
```
bsub -n 20 -W 20:00 -R "rusage[mem=4500,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" ./run.sh
```
-n 20 -> 20 processor cores
-W 20:00 -> set the runtime to max 20 hours !!! HAS TO BE SET ELSE 4h max!!!
-R "rusage[mem=4500]" -> set memory requirement to 4500MB !!!PER CORE!!!
-R "rusage[ngpus_excl_p=8]" -> set the number of gpus to 1
-R "select[gpu_model0==GeForceGTX1080Ti]" -> selects the GPU model 
    -> see available GPUs here: https://scicomp.ethz.ch/wiki/Using_the_batch_system#GPU
-R "rusage[scratch=YYY]" -> if we want to use scratch space on exec node, need to request it too

put files in  /cluster/scratch -> scratch space is not backed up and auto delete is after 2 weeks, max 2.5 TB

#### need to locally install all the additional python files
pip install --user package

#### then also load the installed python_gpu version with the corresponding gcc
module load gcc/8.2.0 python_gpu/3.9.9



bsub -n 6 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=1]" -R "select[gpu_model0==QuadroRTX6000]" < run.sh


----------

## ENV SETUP:

module load gcc/8.2.0 python_gpu/3.9.9

pip install --user pydantic==1.8.2
pip install --user tqdm pytorch-lightning numpy notebook jupyter amrlib penman cached-property unidecode datasets
pip install --user allennlp-models allennlp
pip install --user transformers tokenizers
pip install --user dgl-cu113 dglgo -f https://data.dgl.ai/wheels/dgl_cu113-0.8.1-cp39-cp39-manylinux1_x86_64.whl
pip install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install --user rdflib==4.2.2

#### not sure if we have to do the module load every time..

#### !!!
cd $SCRATCH

#### important download the used RoBERTa model from: https://huggingface.co/roberta-large/tree/main   https://huggingface.co/roberta-base/tree/main
#### then you can set the --model_name_or_path to the path that points to the correct directory and it is fine (issue because we cannot download!)
#### this is to extract and copy all the glue graphs to the correct directories in the glue_data, but you can probably jsut copy it
tar zxvf glue_graphs.tgz
rsync -a glue_graphs/ glue_data/

#### commands I used so far
bsub -n 6 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=1]" -R "select[gpu_model0==QuadroRTX6000]" < run.sh
bsub -n 8 -W 6:00 -R "rusage[mem=3072,ngpus_excl_p=2]" -R "select[gpu_model0==NVIDIATITANRTX]" < run2.sh
bsub -n 20 -W 6:00 -R "rusage[mem=2048]" < run3.sh

#### use bbjobs instead of bjobs!
bbjobs
bjobs -w
bqueues 


##### this is the run3.sh content
#!/usr/bin/bash
TOKENIZERS_PARALLELISM=true
python train.py \
    --do_train --task qnli --data_dir data/glue_data/QNLI --output_dir output_dir3 \
    --model_name_or_path /cluster/scratch/makleine/CSNLP/huggingface_models/roberta-large --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm \
    --gpus 0 --numworkers 6















