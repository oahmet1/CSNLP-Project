# Towards a Deeper Understanding of Semantic Comprehension in Language Models Paired with Semantic Graphs


## Reproducibility Instructions:

### Required packages (euler)

module load gcc/8.2.0 python_gpu/3.9.9

pip install --user pydantic==1.8.2 \
pip install --user tqdm pytorch-lightning numpy notebook jupyter amrlib penman cached-property unidecode datasets \
pip install --user allennlp-models allennlp \
pip install --user transformers tokenizers \
pip install --user dgl-cu113 dglgo -f https://data.dgl.ai/wheels/dgl_cu113-0.8.1-cp39-cp39-manylinux1_x86_64.whl \
pip install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
pip install --user rdflib==4.2.2 \


### required transformers:

possibly need to download the required pretrained transformers from huggingface since the internet connection required by the transformers library may not work on euler

https://huggingface.co/roberta-base/tree/main \
https://huggingface.co/roberta-large/tree/main


### Data


### Graphs


### Training

For training we used the euler_run.sh file which we changed ever so slightly for the different training runs.

Things to change:
- ensure that the output directory does not exist already by adapting the rm -rf to the same dir as the --output_dir flag
- set the --numworkers to the number of dataloader workers (however we noticed that there exists some bug for numworkers != 0 which occurs for the larger datasets where one worker can crash leading to the entire run failing)
- if you want to use the static embededing version of the model set --static_embeddings 1 otherwise set it to 0
- if you want to use the fixed encoder version of the model set --fixed_encoder 1 otherwise set it to 0
- set --formalism to either dm or amr depending on which model you want to train
- set --amr_version 0 to train the model with AMRv0 otherwise set to 1 for AMRv1
- set --task to the desired task (we used qnli, rte and mnli but other glue datasets should also work if the data and graphs exist)
- set --data_dir according to task and relative position of data to euler_run.sh ../../data/glue_data/QNLI
- set --model_name_or_path to the huggingface transformer model name if internet access works otherwise need to point to directory containing the pretrained transformer files available on huggingface.. /cluster/scratch/makleine/CSNLP/models/huggingface_models/roberta_base

NOTE:
- ensure that enough memory is available for qnli and rte about 40GB should be enough for the AMR case, while for the mnli somewhere in the range of 100-120GB of memory are required
- ensure that the GPU has enough memory, for non mnli it should probably be 24GB while for mnli it has to be > 34GB
- the feature file built upon the formalism graphs and the metadata file will only be created if they don't already exist, hence if this code is supposed ot run ensure that the files are deleted!

#### example submission for mnli task with amr. it will take about 12h to train 20 epochs
bsub -G s_stud_infk -W 24:00 -n 5 -R "rusage[mem=24000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=30000]" < euler_run.sh
#### example submission for rte or qnli task with amr. it will take significantly less than 12h to train 20 epochs on those
bsub -G s_stud_infk -W 24:00 -n 5 -R "rusage[mem=10000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=20000]" < euler_run.sh 


### 






# Infusing Finetuning with Semantic Dependencies

The official PyTorch implementation for our paper:

```
@article{wu2021infusing,
    author = {Wu, Zhaofeng and Peng, Hao and Smith, Noah A.},
    title = "{Infusing Finetuning with Semantic Dependencies}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {226-242},
    year = {2021},
    month = {03},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00363},
    url = {https://doi.org/10.1162/tacl\_a\_00363},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00363/1896777/tacl\_a\_00363.pdf},
}
```

## Environment

```bash
pipenv install
```

To install Python 3.9 on Ubuntu use

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install python3.9 python3.9-dev
```

## Pretrained Models

You can find a list of pretrained models, for both SIFT (base + large) and SIFT-Light and for all GLUE tasks (except WNLI), at [this Google Drive folder](https://drive.google.com/drive/folders/1Cz4jrpoYa4w_dY2ZS683C2Bdveddkt6T). You can also run `bash scripts/download_pretrained_base_models.sh` to download all base-sized models at once to `pretrained_base_models/`, which takes around 39G before untar-ing. Most of these models have better performance than the numbers reported in the paper, because the paper reported averages across multiple seeds.

## GLUE Data and Semantic Graphs

Download GLUE data using [this gist](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). We used a prior version of this script -- see the end of this section for some caveats.

```bash
cd data
python download_glue_data.py
```

Both SIFT and SIFT-Light require the target dataset to be pre-parsed into semantic graphs. The instructions can be found in [decode_che_et_al.md](decode_che_et_al.md). However, because it requires the use of the CoNLL 2019 pipeline to generate the companion data (see [decode_che_et_al.md](decode_che_et_al.md)), which is not yet public accessible as far as we know, we release pre-parsed semantic graphs for all GLUE tasks except WNLI [here](https://drive.google.com/file/d/1RQu8fbfRF7ne6JttsqxhoPrin9zvBFB1/view) (13GB before untar-ing). We provide graphs in all semantic formalisms in CoNLL 2019 except AMR (i.e., DM, EDS, PSD, UCCA), because the unanchored nature of AMR makes it theoretically impossible to do the wordpiece-node alignment. These formalisms perform similarly in our preliminary experiments and we only reported the numbers with DM in the paper.

The semantic graphs need to be in the same directories as the original datasets. You can do something like this:

```bash
# Assuming the current directory contains glue_data/ and glue_graphs.tgz
tar xzvf glue_graphs.tgz
for dataset in $(ls glue_graphs); do echo ${dataset}; cp glue_graphs/${dataset}/* glue_data/${dataset}/; done
```

Both SIFT and SIFT-Light require the _entire_ dataset (i.e., train, dev, and test) to be pre-parsed into semantic graphs. Some of this is not needed conceptually. For example, SIFT shouldn't need the training graphs for evaluation or inference, and SIFT-Light shouldn't need _any_ semantic graphs in non-training modes. However, we require these for an easier implementation. There can be implementations that do not require this information.

Note that during the first time that you use a particular dataset (for either training, evaluation, or inference), two cached data files are created in the dataset directory for faster data loading later. This could be CPU memory intensive for large datasets. For example, processing QQP for the first time can require around 200-300GB of CPU memory and a few hours of processing time. Subsequent training runs should be much less resource-intensive.

We used a [prior version of the GLUE downloading script](https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py), but the dataset files linked there have expired and been replaced with new locations. We believe the datasets should remain the same and our pre-parsed graphs still applicable. The only exception we are aware of so far is that three malformed question pairs in QQP have been removed in the new version. Therefore, you need to manually remove the graphs corresponding to the following (0-based) indices from both the `.rdf` and `.metadata` files:

```
497656  n/a
497657  My Chinese name is Haichao Yu. What English name is most suitable for me considering the pronounciation of my Chinese name?
...
620242  How can I create an Android app?
620243
...
724450  How can I develop android app?
724451
```

## Evaluation/Inference With Pretrained Models

The pretrained models can be directly evaluated (on the dev set) with the following command, provided that the dataset and semantic graphs are in place following the previous section. `${MODEL_DIR}` is the directory to the model, e.g. `pretrained_models/CoLA_SIFT_base`. `${DATA_DIR}` is the directory to the dataset, e.g. `data/glue_data/CoLA`.

```bash
python evaluate.py --model_dir ${MODEL_DIR} --override_data_dir ${DATA_DIR}
```

You should get the following numbers:

|                 | SIFT-base | SIFT-Light-base | SIFT-large |
| --------------- | --------- | --------------- | ---------- |
| CoLA            | 65.80     | 65.78           | 71.84      |
| MRPC            | 90.69     | 90.93           | 92.16      |
| RTE             | 81.95     | 81.95           | 89.17      |
| SST-2           | 95.64     | 94.84           | 96.67      |
| STS-B           | 91.50     | 91.23           | 92.71      |
| QNLI            | 93.39     | 93.10           | 94.89      |
| QQP             | 91.96     | 91.75           | 92.29      |
| MNLI-matched    | 88.07     | 87.74           | 90.55      |
| MNLI-mismatched | 87.66     | 87.57           | 90.37      |

Similarly, the pretrained models can be directly used for inference on the test set:

```bash
python inference.py --model_dir ${MODEL_DIR} --override_data_dir ${DATA_DIR}
```

You can also use the pretrained models for inference on other tasks with the following command. If it is a new task, you will need to modify `data_readers/__init__.py` and `metrics.py` to add data reading logic and the metric information. The original training data directory still needs to be specified with corresponding semantic graphs for engineering simplification.

```bash
python inference.py --model_dir ${MODEL_DIR} --override_data_dir ${DATA_DIR} --inference_task ${TASK_NAME} --inference_data_dir ${INFERENCE_DATA_DIR}
```

You will see similar runtime and memory overhead with both SIFT and SIFT-Light. This is because we are not doing anything special to remove the RGCN layers from SIFT-Light in a non-training mode, but we are not using their output. This is, again, to simplify the code.

## Training Your Own Models

An example command to train CoLA is given below. CoLA is known for high variance, so anything between 62 to 66 best dev MCC is probably normal.

```bash
python train.py \
    --do_train --task cola --data_dir data/glue_data/CoLA --output_dir output_dir \
    --model_name_or_path roberta-base --max_seq_length 256 \
    --num_train_epochs 20 --train_batch_size 8 --gradient_accumulation_steps 4 --eval_batch_size 16 \
    --learning_rate 2e-5 --warmup_ratio 0.06 --weight_decay 0.1 \
    --formalism dm --n_graph_layers 2 --n_graph_attn_composition_layers 2 \
    --graph_n_bases 80 --graph_dim 512 --post_combination_layernorm
```
