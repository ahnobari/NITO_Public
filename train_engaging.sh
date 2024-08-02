#!/bin/bash

#SBATCH -n 200
#SBATCH -p pi_faez
#SBATCH --gres=gpu:4
#SBATCH --mem=2000000
#SBATCH -t 4-00:00
#SBATCH -o log.out

if [ -z ${NITO_Field_hidden_size+x} ]; then export NITO_Field_hidden_size=26500; else echo "NITO_Field_hidden_size is set to '$NITO_Field_hidden_size'"; fi
if [ -z ${NITO_Field_n_layers+x} ]; then export NITO_Field_n_layers=8; else echo "NITO_Field_n_layers is set to '$NITO_Field_n_layers'"; fi
if [ -z ${NITO_BC_emb_size+x} ]; then export NITO_BC_emb_size=170; else echo "NITO_BC_emb_size is set to '$NITO_BC_emb_size'"; fi
if [ -z ${NITO_C_mapping_size+x} ]; then export NITO_C_mapping_size=512; else echo "NITO_C_mapping_size is set to '$NITO_C_mapping_size'"; fi
if [ -z ${NITO_samples+x} ]; then export NITO_samples=64; else echo "NITO_samples is set to '$NITO_samples'"; fi
if [ -z ${NITO_batch_size+x} ]; then export NITO_batch_size=4; else echo "NITO_batch_size is set to '$NITO_batch_size'"; fi

module load cuda/12.4.0-x86_64

export PATH=/home/ahnobari/miniforge3/bin:${PATH}
source activate DDP

torchrun --nproc_per_node=4 train.py --multi_gpu --mixed_precision \
    --Field_hidden_size $NITO_Field_hidden_size \
    --Field_n_layers $NITO_Field_n_layers \
    --BC_emb_size $NITO_BC_emb_size \
    --C_mapping_size $NITO_C_mapping_size \
    --samples $NITO_samples \
    --batch_size $NITO_batch_size \
    --DDP
# torchrun --nproc_per_node=4 train.py --multi_gpu --mixed_precision --Field_hidden_size 26500 --Field_n_layers 8 --BC_emb_size 170 --C_mapping_size 512 --samples 64 --batch_size 4 --DDP

# python train.py --multi_gpu --mixed_precision --Field_hidden_size 25000 --Field_n_layers 8 --BC_emb_size 128 --C_mapping_size 512 --samples 512 --batch_size 16
