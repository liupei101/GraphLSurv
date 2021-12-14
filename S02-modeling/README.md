# S02: Survival Modeling

## Model training

After you finished S01-preprocessing (please ensure that you have followed the given instructions), you can train GraphLSurv on the dataset NLST by the command:

```bash
# build the initial KNN graph for each patient from NLST
cd ./scripts
./S1-Build-Graph.sh

# train GraphSurv model on NLST
cd ..
python3 main.py --config config/report-nlst.yml --multi_run 
``` 

## Data splitting

All methods proposed earlier than our GraphLSurv did not make their training/val/test sets publicly-available. It's really tough for one who proposes an novel method to make a fair comparison with previous methods. For this reason, we reproduce all previous methods on TCGA-BRCA, and intentionally use the same data splitting in model training and evaluation.

Therefore, we provide all `patient_id` of training/val/test sets from NLST and TCGA-BRCA. Hope it could be used as benchmarks for the further research of methodology. 

### A brief introduction

`DS` may be `nlst` or `tcga_brca`. The file from `data_split/DS/DS-data-split-seed42.npz` contains the patient ID from training/val/test sets. It is obtained by:
- firstly read a list of patients from table `data_split/DS/DS_path_full.csv`
- then randomly split the list into three parts as training/val/test set by the function `utils.DataSplit` with the random seed of 42
- finally pack and save them as a npz file
