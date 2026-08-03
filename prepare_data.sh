python prepare_data.py \
    --test_split_pct 0.005 \
    --context_length 1024 \
    --path_to_data_store "./prepped_data" \
    --dataset_split_seed 42 \
    --num_workers 8 \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --num_c4_shards 1
