#!/bin/bash
set -e

# If running at test level
IF_TEST=0
# Directory storing tiles generated from step2
# IT MUST BE UNDER DIR_DATA or DIR_DATA_EXP
DIR_NAME=tiles-l1-s256

# Sampling method: random_be = random by energy
SM=random_be
# Sampling size
SS=1000
# Sampling pool size
PS=1000

# Directory name for storing sampled patches
# IT WILL BE CREATED UNDER DIR_DATA or DIR_DATA_EXP 
DIR_FEAT=feats-l1-s256-m${SM}-n${SS}-no_color_norm

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
    CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py \
        --data_h5_dir ${DIR_DATA_EXP}/${DIR_NAME} \
        --data_slide_dir ${DIR_DATA_EXP}/${DIR_RAW} \
        --csv_path ${DIR_DATA_EXP}/${DIR_NAME}/process_list_autogen.csv \
        --feat_dir ${DIR_DATA_EXP}/${DIR_FEAT} \
        --batch_size 128 \
        --slide_ext .svs \
        --sampler ${SM} \
        --sampler_size ${SS} \
        --sampler_pool_size ${PS} \
        --color_norm \
        --no_auto_skip
else
    echo "running for extracting features from all tiles"
    CUDA_VISIBLE_DEVICES=0 python3 extract_features_fp.py \
        --data_h5_dir ${DIR_DATA}/${DIR_NAME} \
        --data_slide_dir ${DIR_DATA}/${DIR_RAW} \
        --csv_path ${DIR_DATA}/${DIR_NAME}/process_list_autogen.csv \
        --feat_dir ${DIR_DATA}/${DIR_FEAT} \
        --batch_size 64 \
        --slide_ext .svs \
        --sampler ${SM} \
        --sampler_size ${SS} \
        --sampler_pool_size ${PS} \
        --no_auto_skip --slide_in_child_dir
fi
