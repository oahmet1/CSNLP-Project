#!/usr/bin/bash

module load gcc/8.2.0 python_gpu/3.9.9
export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPs_PROXY=http://proxy.ethz.ch:3128

python dataset_to_amr.py rte


# mnli cola mnli_matched mnli_mismatched mrpc qnli sst2 qqp stsb wnli ax rte
