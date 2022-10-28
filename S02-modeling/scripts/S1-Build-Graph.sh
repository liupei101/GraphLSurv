#!/bin/bash
set -e

# If running at test level
IF_TEST=0
# Directory storing tiles sampled from S01-preprocessing/scripts/4-Extracting-Patches-Features.sh
# IT MUST BE UNDER DIR_DATA or DIR_DATA_EXP
DIR_FEAT=feats-l1-s256-mrandom_be-n1000-color_norm

# Build initial KNN graph
method=knn
K=6
T=0
# The output directory UNDER ${DIR_DATA}/${DIR_FEAT} or ${DIR_DATA_EXP}/${DIR_FEAT}
DIR_GRAPH=graph-${method}-k${K}-t${T}

# Path of CLAM Library
DIR_REPO=../tools

# Path of pathology images, only used for tuning experiments
DIR_DATA_EXP=/home/liup/tmp/NLST_Path_Tmp
# Path of pathology images, only used for formal experiments
DIR_DATA=/NAS/Dataset/NLST/PathologySlide
# tables directory
TBS_DATA=/NAS/Dataset/NLST/Pathology


cd ${DIR_REPO}

if [ ${IF_TEST} -eq 1 ]; then
    echo "running for test"
    python3 graph_builder.py \
        --dir_input ${DIR_DATA_EXP}/${DIR_FEAT}/h5_files \
        --dir_output ${DIR_DATA_EXP}/${DIR_FEAT}/${DIR_GRAPH} \
        --csv_sld2pat ${TBS_DATA}/nlst_path_path2pat.csv \
        --graph_level patient \
        --num_neighbours ${K} \
        --threshold ${T} \
        --num_workers 1 \
        --verbose
else
    echo "running for building graphs of all slides"
    python3 graph_builder.py \
        --dir_input ${DIR_DATA}/${DIR_FEAT}/h5_files \
        --dir_output ${DIR_DATA}/${DIR_FEAT}/${DIR_GRAPH} \
        --csv_sld2pat ${TBS_DATA}/nlst_path_path2pat.csv \
        --graph_level patient \
        --method ${method} \
        --num_neighbours ${K} \
        --threshold ${T} \
        --num_workers 2 \
        --verbose
fi
