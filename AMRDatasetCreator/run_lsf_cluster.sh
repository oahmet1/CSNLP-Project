#!/usr/bin/bash

module load gcc/8.2.0 python_gpu/3.9.9

python dataset_to_amr.py mnli


# mnli cola mnli_matched mnli_mismatched mrpc qnli sst2 qqp stsb wnli ax rte