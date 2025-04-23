This is the beginning of how to modify a LLama-2-7B based on the Hugging Face modeling.

The evaluation framework is from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). And I'll show you how to modify a model (e.g. llama2-7B) for evaluation.

Find modeling.py here:

```
lm-evaluation-harness/
└── lm_eval/
    └── models/
        ├── __init__.py
        ├── anthropic_llms.py
        ├── api_models.py
        ├── configuration_deepseek.py
        ├── modeling_deepseek.py  ← Implements DeepSeek model integration
        └── ...
```

Env setting:
```
conda create -n llm python==3.9
conda activate llm
cd lm-evaluation-harness
pip install -e .
```
