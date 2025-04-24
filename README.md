This is the beginning of how to modify a LLama-2-7B based on the Hugging Face modeling.

The evaluation framework is from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). And I'll show you how to modify a model (e.g. llama2-7B) for evaluation.

0. Huggingface Framework:
Usually, opensource models' parameters are stored in the model card of huggingface. For example, when using [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main), the code structure is shown below:

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
1. Env setting:
   
```
conda create -n llm python==3.9
conda activate llm
pip install importlib_metadata
pip install SentencePiece
pip install protobuf
cd lm-evaluation-harness
pip install -e .
```

2. Find modeling.py here:

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
3. Command line

You can change any llama model by replacing the path of the hugging face model card. MMLU and Hellaswag are two evaluation datasets.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True \
    --tasks mmlu,hellaswag \
    --batch_size 4
```
