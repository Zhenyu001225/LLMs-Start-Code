CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True \
    --tasks mmlu,hellaswag \
    --batch_size 4