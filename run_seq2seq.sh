#!/bin/bash
source /applis/environments/cuda_env.sh dahu 10.1
source /applis/environments/conda.sh
conda activate TF-GPU
cd /bettik/PROJECTS/pr-serveurgestuel/src/SLT/S2T
python S2T_MA.py
exit 0
