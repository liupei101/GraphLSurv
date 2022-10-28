#!/bin/bash
set -e

# If running at tuning level
IF_TUNING=0

# Sample patches of SIZE x SIZE at LEVEL 
LEVEL=1
SIZE=256

# Path of CLAM Library
DIR_REPO=../tools/CLAM

# Base path of pathology images, only used for tuning experiments
DIR_DATA_EXP=/home/liup/tmp/NLST_Path_Tmp
# Base path of pathology images, only used for formal experiments
DIR_DATA=/NAS1/Dataset/NLST/PathologySlide

# Directory storing raw slides (files ended with .svs or others) 
# MUST BE UNDER BASE PATH specified above 
DIR_RAW=raw


cd ${DIR_REPO}

if [ ${IF_TUNING} -eq 1 ]; then
    echo "run for tuning parameters"
    python3 create_patches_fp.py \
        --source ${DIR_DATA_EXP}/${DIR_RAW} \
        --save_dir ${DIR_DATA_EXP}/tiles-l${LEVEL}-s${SIZE} \
        --patch_size ${SIZE} \
        --step_size ${SIZE} \
        --preset nlst.csv \
        --patch_level ${LEVEL} \
        --seg --patch --stitch \
        --no_auto_skip
else
    echo "run for seg&tile to all images"
    CUDA_VISIBLE_DEVICES=1 python3 create_patches_fp.py \
        --source ${DIR_DATA}/${DIR_RAW} \
        --save_dir ${DIR_DATA}/tiles-l${LEVEL}-s${SIZE} \
        --patch_size ${SIZE} \
        --step_size ${SIZE} \
        --preset nlst.csv \
        --patch_level ${LEVEL} \
        --seg --patch --stitch \
        --no_auto_skip --in_child_dir
fi
