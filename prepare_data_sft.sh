python prepare_data_sft.py \
    --test_split_pct 0.01 \
    --context_length 1024 \
    --path_to_data_store "./prepped_sft_data" \
    --dataset_split_seed 42 \
    --num_workers 8 \
    --hf_model_name "answerdotai/ModernBERT-base"
