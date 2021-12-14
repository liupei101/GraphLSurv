# GraphLSurv 
Repo of source code of a paper "GraphLSurv: Adaptive Structure Learning of Histopathological Whole-Slide Images for Survival Prediction". (Submitted to TMI, under-review)

**Note** that current version is only for peer-reviewing. More details of this project will be published in this repo upon acceptance. Questions or communications are welcomed.

## WSI Preprocessing

See more details in **S01-preprocessing**.

## GraphLSurv Modeling

See more details in **S02-modeling**.

## Experiments Reproducing

We have reproduced methods of deepConvSurv, WSISA, DeepGraphSurv and DeepAttnMISL on NLST and TCGA-BRCA.

Source codes are from official release:
- deepConvSurv & WSISA: https://github.com/uta-smile/WSISA
- DeepGraphSurv: https://github.com/uta-smile/DeepGraphSurv
- DeepAttnMISL: https://github.com/uta-smile/DeepAttnMISL

All experimental details including source code and running log could be found at [here](https://drive.google.com/drive/folders/1v_YbJiuxKMMGN-ybeW1iQOi2F-sRw0mo?usp=sharing).

## Experimental results

### Overall performance

Both Concordance index (CI) and AUC are evaluated on NLST and TCGA_BRCA.

Also, Prognostic power is evaluated by comparing the differences between high- and low- risk group.

### Abalations

Ablations include:
- Adaptive graph's effects on the model
- Exploration to adaptive graphs with different connection strength.
- Model architecture: mix GCN-HMP or pure GCN-HMP

### Visualization

Adaptive graph visualization on two WSIs.

