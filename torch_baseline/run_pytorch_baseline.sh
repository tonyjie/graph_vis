#!/bin/bash
# Usage: bash run_pytorch_baseline.sh DRB1-3123 1000
CHRM_NAME=$1 # DRB1-3123, mhc
BATCH_SIZE=$2
HOME_DIR=/home/jl4257/Project/Pangenomics/graph_vis/data_pangenome/${CHRM_NAME}
OG_FILE=${HOME_DIR}/${CHRM_NAME}.og
set -x
commands="env CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=${HOME}/odgi_niklas/lib python odgi_real_batch_sgd.py ${OG_FILE} --num_iter 30 --batch_size ${BATCH_SIZE} --cuda --log_interval 2000 --lay"
WORK_DIR=${HOME_DIR}/batch_size=${BATCH_SIZE}
mkdir -p ${WORK_DIR}
LOG_FILE=${WORK_DIR}/log.txt
if [ -f ${LOG_FILE} ]; then
    rm ${LOG_FILE}
fi

${commands} 2>&1 | tee -a ${LOG_FILE}

