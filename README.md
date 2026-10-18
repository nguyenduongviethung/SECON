# SECON


Datasets and Baselines from https://github.com/microsoft/CodeBERT

## Generating summaries

`generate_summary.py` accepts a custom prompt with `--prompt`. Use `{language}`
and `{code}` as placeholders for each input record:

```powershell
python generate_summary.py --input input.jsonl --output summaries.jsonl --model_path path\to\model --prompt "Summarize this {language} code: {code}"
```

If `--prompt` is omitted, the built-in summary prompt is used.

# Reference
If you find our source useful, please consider citing our work.

Zhang X, Lin Z, Hu X, et al. SECON: Maintaining Semantic Consistency in Data Augmentation for Code Search[J]. ACM Transactions on Information Systems, 2025.
