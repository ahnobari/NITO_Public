# Run this script to run all of the training needed for the experiments in the paper
# Train NITO on 64x64 Data
echo "Training NITO on 64x64 Data ..."
python train.py \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --mixed_precision \
       --compile \
       --Optimizer AdamW \
       --shape_normalize \
       --batch_size 64 \
       --samples 1024 \
       --epochs 50 \
       --data ./Data \
       --checkpoint_dir ./Checpoints/64x64 \
       --start_index 0 --end_index 47000

# Train NITO on 256x256 Data
echo "Training NITO on 256x256 Data ..."
python train.py \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --mixed_precision \
       --compile \
       --Optimizer AdamW \
       --shape_normalize \
       --batch_size 16 \
       --samples 1024 \
       --epochs 50 \
       --data ./Data \
       --checkpoint_dir ./Checpoints/256x256 \
       --start_index 47000 --end_index 107000

# Train NITO on 64x64 and 256x256 Data
echo "Training NITO on 64x64 and 256x256 Data ..."
python train.py \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --mixed_precision \
       --compile \
       --Optimizer AdamW \
       --shape_normalize \
       --batch_size 16 \
       --samples 1024 \
       --epochs 50 \
       --data ./Data \
       --checkpoint_dir ./Checpoints/64x64_256x256 \
       --start_index 0 --end_index 107000

# Train NITO on All Data
echo "Training NITO on All Data ..."
python train.py \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --mixed_precision \
       --compile \
       --Optimizer AdamW \
       --shape_normalize \
       --batch_size 16 \
       --samples 1024 \
       --epochs 50 \
       --data ./Data \
       --checkpoint_dir ./Checpoints/All

