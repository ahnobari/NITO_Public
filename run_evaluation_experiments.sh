# Run this script to run all of the evaluations needed for the experiments in the paper

echo "Running 64x64 on NITO trained on all data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/All/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --end_index 1000 \
       --precompute_kernel \
       --ignore_outliers \
       --time_inference \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --results_dir ./Results/64x64__All
    
echo "Running 64x64 on NITO trained on 64x64 data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/64x64/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --end_index 1000 \
       --precompute_kernel \
       --ignore_outliers \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --results_dir ./Results/64x64__64x64

echo "Running 64x64 on NITO trained on 256x256 data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/256x256/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --end_index 1000 \
       --precompute_kernel \
       --ignore_outliers \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --results_dir ./Results/64x64__256x256

echo "Running 64x64 on NITO trained on 64x64 and 256x256 data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/64x64_256x256/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --end_index 1000 \
       --precompute_kernel \
       --ignore_outliers \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --results_dir ./Results/64x64__64x64_256x256

echo "Running 64x48 on NITO trained on all data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/All/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 3000 --end_index 4000 \
       --precompute_kernel \
       --ignore_outliers \
       --time_inference \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/64x48__All

echo "Running 64x32 on NITO trained on all data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/All/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 2000 --end_index 3000 \
       --precompute_kernel \
       --ignore_outliers \
       --time_inference \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/64x32__All

echo "Running 64x16 on NITO trained on all data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/All/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 4000 --end_index 5000 \
       --precompute_kernel \
       --ignore_outliers \
       --time_inference \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/64x16__All

echo "Running 256x256 on NITO trained on all data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/All/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 1000 --end_index 2000 \
       --precompute_kernel \
       --ignore_outliers \
       --time_inference \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/256x256__All

echo "Running 256x256 on NITO trained on 64x64 data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/64x64/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 1000 --end_index 2000 \
       --precompute_kernel \
       --ignore_outliers \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/256x256__64x64

echo "Running 256x256 on NITO trained on 256x256 data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/256x256/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 1000 --end_index 2000 \
       --precompute_kernel \
       --ignore_outliers \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/256x256__256x256

echo "Running 256x256 on NITO trained on 64x64 and 256x256 data"
python evaluate.py \
       --data ./Data/Test \
       --batch_size 5 \
       --checkpoint ./Checkpoints/64x64_256x256/checkpoint_epoch_50.pth \
       --Field_hidden_size 1024 \
       --BC_emb_size 80 \
       --start_index 1000 --end_index 2000 \
       --precompute_kernel \
       --ignore_outliers \
       --mixed_precision \
       --compile \
       --shape_normalize \
       --supress_warnings \
       --save_raw --save_intermediate --save_optimized --save_CE --save_VFE \
       --multi_processing \
       --results_dir ./Results/256x256__64x64_256x256


