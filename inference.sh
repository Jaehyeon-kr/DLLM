python inference.py \
    --safetensors_path "./dllm_exp/dllm_sft/final_model/model.safetensors" \
    --seq_len 64 \
    --num_steps 64 \
    --strategy low_confidence \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --prompt "whiat is dog?"
