This guide will show you how to modify and evaluate a LLaMA-2-7B model from [HuggingFace](https://huggingface.co/) using the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.

0. HuggingFace Model Format:
Most open-source models store their parameters and configuration files on Hugging Face. For example, [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) model repository includes the following structure:

```
Llama-2-7b-hf/
├── .gitattributes                     
├── LICENSE.txt                       
├── README.md                          
├── Responsible-Use-Guide.pdf          
├── USE_POLICY.md                      
│
├── config.json                        # Hugging Face model config (e.g., hidden size, layers)
├── generation_config.json             # Generation defaults (e.g., max_length, temperature)
│
├── model-00001-of-00002.safetensors   # First shard of model weights in safetensors format
├── model-00002-of-00002.safetensors   # Second shard of model weights in safetensors format
├── model.safetensors.index.json       # Index file to map layers to shards
│
├── pytorch_model-00001-of-00002.bin   # (Optional) PyTorch format weight shard
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json       # Index for PyTorch binary shards
│
├── special_tokens_map.json            # Mapping of special tokens like <bos>, <eos>
├── tokenizer.json                     # Tokenizer vocabulary and merges (Hugging Face format)
├── tokenizer.model                    # SentencePiece model
└── tokenizer_config.json              # Tokenizer config (e.g., bos_token, eos_token)
```
1.  Environment Setup:

To set up the evaluation environment:

```
conda create -n llm python==3.9
conda activate llm
pip install importlib_metadata
pip install SentencePiece
pip install protobuf
cd lm-evaluation-harness
pip install -e .
```

2. Find modeling.py here.

To modify your LlamA model, navigate to:

```
lm-evaluation-harness/
└── lm_eval/
    └── models/
        ├── __init__.py
        ├── anthropic_llms.py
        ├── api_models.py
        ├── modeling_llama.py  ← Implements llama model integration
        └── ...
```
3. Run evaluation.

To evaluate the model on benchmarks like MMLU and Hellaswag, use the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True \
    --tasks mmlu,hellaswag \
    --batch_size 4
```
This command is in lm_eval.sh
