#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

dist_metric=$1

python3 scripts/analyze_variants.py --distance_metric $dist_metric
