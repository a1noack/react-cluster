# Detecting and characterizing unseen attacks
## High level description of scripts and their associated jobs
 - `train_siamese.py` and the associated SLURM job files in `jobs/siamese/` allow one to train a Siamese network to embed textual attacks (extracted using the REACT feature extraction pipeline) 
 - `eval_siamese.py` and the associated SLURM job files in `jobs/eval` allow one to evaluate a trained Siamese network on the task of novelty prediction using: (1) distance to the known attack mean embeddings, (2) Local Outlier Factor in the Siamese network embedding space. These also run the  same/different novel attack prediction experiments where the task is to determine if two groups were created by the same novel attack method or not.
 - `train_clf.py` and the associated SLURM job files in `jobs/classify` allow one to train a boosted tree classifier for attack labeling and then evaluate the trained classifier on the task of novelty prediction using the output entropy of the classifier
 - `analyze_variants.py` and the associated SLURM job files in `jobs/variants` allow for one to determine how similar/different variant attacks are to/from their original attack methods
 -  `cluster.py` and the associated SLURM job files in `jobs/cluster` allow one to cluster attacks using various clustering algorithms such as k-means. This script has not been updated in a while and is close to being deprecated.
## Setup
If you do not already have a conda environment with Python 3.8 installed, create a new conda environment with 3.8. Now activate this environment.

Clone this repository and `cd` into it.

In this conda environment, run `$ pip install -r requirements.txt` in order to install the required packages.

To run any of the scripts in this repository, you will need to download the joblib files that hold the extracted features for the attacks. They should be in [the shared Google Drive folder](https://drive.google.com/drive/folders/1978mX878T-B6P23s-YGzsGSsoeim5xp7?usp=sharing). Create this directory structure `react-cluster/data/extracted_features/` and then place the downloaded joblib files into the `extracted_features/` subdirectory.

To run the any of the SLURM job files in `jobs/*/`, you will have to create the following three directories: `react-cluster/jobs/errors/`, `react-cluster/jobs/logs/`, and `react-cluster/output/`.

If running `analyze_variants.py` you will need to download `whole_dataset_with_meta.csv`, and if you are running `cluster.py` you will also need to download `whole_catted_dataset.csv` from [the shared Google Drive folder](https://drive.google.com/drive/folders/1WcveERnyP1kA1_9tHNyLgoTbBkGPHkju?usp=sharing). Create this directory structure `react-cluster/data/original_data/` and then place the downloaded CSV files into the `original_data/` subdirectory. You will also have to download the actual text produced by the variants which can be found by navigating down the Google Drive directory structure starting [here](https://drive.google.com/drive/folders/11LRMS9ma1FKEnlYBbv5KtiiaNvLTTY51?usp=sharing).
