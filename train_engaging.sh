#!/bin/bash

#SBATCH -n 100
#SBATCH -p pi_faez
#SBATCH --gres=gpu:4
#SBATCH --mem=1000000
#SBATCH -t 4-00:00

module load cuda/12.4.0-x86_64

export PATH=/home/ahnobari/miniforge3/bin:${PATH}
source activate topology

python train.py --multi_gpu --mixed_precision --Field_hidden_size 27000 --Field_n_layers 8 --BC_emb_size 128 --C_mapping_size 512 --samples 512 --batch_size 24