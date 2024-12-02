CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=deepseek-ai/deepseek-moe-16b-base,trust_remote_code=True \
    --tasks mmlu,hellaswag \
    --batch_size 4