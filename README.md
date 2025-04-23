This is the beginning of how to modify a LLama-2-7B based on the Hugging Face modeling.

The evaluation framework is from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). And I'll show you how to modify a model (e.g. llama2-7B) for evaluation.

Code Structure:



Env setting:
```
conda create -n llm python==3.9
conda activate llm
cd lm-evaluation-harness
pip install -e .
```
