#!/bin/bash
set -e

# If running at test level
IF_TEST=0

# Directory storing tiles generated from step2
# IT IS UNDER PATH DIR_DATA or DIR_DATA_EXP
DIR_NAME=tiles-l1-s256

# Path of CLAM Library
DIR_REPO=../tools/CLAM

# Path of pathology images, only used for tuning experiments
DIR_DATA_EXP=/home/liup/tmp/NLST_Path_Tmp
# Path of pathology images, only used for formal experiments
DIR_DATA=/NAS1/Dataset/NLST/PathologySlide

# Directory storing raw slides (files ended with .svs or others) 
# MUST BE UNDER BASE PATH specified above 
DIR_RAW=raw


cd ${DIR_REPO}

if [ ${IF_TEST} -eq 1 ]; then
    echo "running for test"
    CUDA_VISIBLE_DEVICES=1 python3 calculate_patches_energy.py \
        --data_h5_dir ${DIR_DATA_EXP}/${DIR_NAME}/patches \
        --data_slide_dir ${DIR_DATA_EXP}/${DIR_RAW} \
        --csv_path ${DIR_DATA_EXP}/${DIR_NAME}/process_list_autogen.csv \
        --out_h5_dir ${DIR_DATA_EXP}/${DIR_NAME}/patches \
        --num_workers 1
else
    echo "running for calculating image energy of all tiles"
    CUDA_VISIBLE_DEVICES=1 python3 calculate_patches_energy.py \
        --data_h5_dir ${DIR_DATA}/${DIR_NAME}/patches \
        --data_slide_dir ${DIR_DATA}/${DIR_RAW} \
        --csv_path ${DIR_DATA}/${DIR_NAME}/process_list_autogen.csv \
        --out_h5_dir ${DIR_DATA}/${DIR_NAME}/patches \
        --num_workers 1 \
        --slide_in_child_dir --auto_skip
fi
