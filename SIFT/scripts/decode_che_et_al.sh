#!/bin/bash
# This scripts takes the sentences .lines file and the annotated .conllu file of a dataset
# and generates the semantic graphs .rdf and associated metadata .metadata
# Assumes the presence of ${DATASET_DIR}/{train,dev,test}.{lines,conllu}
# Make sure the Python environment has
# - allennlp==0.9.0 (note this is different from our main requirement!)
# - an appropriate cuda-enabled PyTorch version
# - rdflib
# And $CUDA_VISIBLE_DEVICES is set

set -e

if [[ ${#} -ne 2 ]]; then
  echo "usage: scripts/decode_che_et_al.sh DATASET_DIR SPLIT"
  exit 1;
fi

PROJECT_DIR=$(pwd)
DATASET_DIR=$(cd $1; pwd)
DATASET_NAME=$2


echo ${DATASET_DIR}

N_VISIBLE_GPUS=`nvidia-smi -L | wc -l`


cd ${PROJECT_DIR}/..
echo "Converting to UDPipe"
# The resulting file is equivalent to the `evaluation/udpipe.mrp` in the original CoNLL-19 release.
# Note that this steps removes sentences that are empty; we add them back in mrp_to_rdf.py
python mtool/main.py --read conllu --write mrp --text ${DATASET_DIR}/${DATASET_NAME}.lines ${DATASET_DIR}/${DATASET_NAME}.conllu ${DATASET_DIR}/${DATASET_NAME}.udpipe.mrp
cd ${PROJECT_DIR}
echo "Converting to MRP"
# The resulting file is equivalent to the `evaluation/input.mrp` in the original CoNLL-19 release.
python scripts/udpipe_to_mrp.py ${DATASET_DIR}/${DATASET_NAME}.udpipe.mrp ${DATASET_DIR}/${DATASET_NAME}.mrp
cd ${PROJECT_DIR}/..
echo "Preprocessing"
rm -rf ${DATASET_DIR}/${DATASET_NAME}_aug_mrp
mkdir ${DATASET_DIR}/${DATASET_NAME}_aug_mrp
python HIT-SCIR-CoNLL2019/toolkit/preprocess_eval.py ${DATASET_DIR}/${DATASET_NAME}.udpipe.mrp ${DATASET_DIR}/${DATASET_NAME}.mrp --outdir ${DATASET_DIR}/${DATASET_NAME}_aug_mrp
echo "Preprocessed successfully"

# We parse all formalisms in parallel. If we have >= 4 GPUs, parse on separate GPUs.
cd ${PROJECT_DIR}/../HIT-SCIR-CoNLL2019
GPU_IDX=0
for FORMALISM in dm psd eds ucca; do
    case $FORMALISM in

    dm | psd)
        predictor_class="transition_predictor_sdp"
        ;;

    eds)
        predictor_class="transition_predictor_eds"
        ;;

    ucca)
        predictor_class="transition_predictor_ucca"
        ;;
    esac

    echo "Parsing ${FORMALISM}"
    allennlp predict \
        --cuda-device $GPU_IDX \
        --batch-size 32 \
        --output-file ${DATASET_DIR}/${DATASET_NAME}.${FORMALISM}.mrp \
        --predictor ${predictor_class} \
        --include-package utils \
        --include-package modules \
        ../HIT-SCIR-CoNLL2019-model/${FORMALISM}/${FORMALISM}.tar.gz \
        ${DATASET_DIR}/${DATASET_NAME}_aug_mrp/${FORMALISM}.mrp &

    if [[ $N_VISIBLE_GPUS -ge 4 ]]; then
        ((GPU_IDX=GPU_IDX+1))
    fi
done

wait

cd ${PROJECT_DIR}
echo "Converting to rdf"
for FORMALISM in dm psd eds ucca; do
    python scripts/mrp_to_rdf.py ${DATASET_DIR}/${DATASET_NAME}.${FORMALISM}.mrp ${DATASET_DIR}/${DATASET_NAME}.${FORMALISM}.rdf ${DATASET_DIR}/${DATASET_NAME}.${FORMALISM}.metadata
done


echo "All done!"
