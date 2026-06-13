# Leakage-aware eye-movement workflow: open-source-safe code

This package contains cleaned, repository-ready code extracted from the shareable parts of the working scripts for the BRM submission.

## What was added from the second code file

The second code file contained corresponding workflow code for:

- participant/background summaries;
- grouping token-level rows into trial-level list-valued rows and expanding them back to token-level rows;
- missing-value handling and lexical-feature merging;
- TEXT / READER / RXT domain-separated splits;
- split-aware n-gram surprisal;
- reader-level metadata permutation and CKA diagnostics;
- residual small-sample reader calibration;
- LLM prompt generation and local Ollama inference;
- parsing/evaluating LLM numeric outputs with token-level and group-level metrics.

These components have been rewritten into configurable modules with no private paths, no raw data, no credentials, and no `eval()`.

## Excluded for safety or irrelevance

- OpenAI API calls containing a hard-coded key;
- web scraping / Selenium utilities;
- unrelated metaphor, image, LoRA, and grammar-type experiments;
- local absolute model paths and GPU-machine-specific code;
- one-off plotting scripts with hard-coded private data values;
- code that requires undistributable raw CECO files.

## Basic use

```bash
pip install -r requirements.txt
```

Create split labels:

```python
from brm_eye_workflow.splits import make_domain_splits
make_domain_splits("data/token_level.csv", "outputs/token_level_with_splits.csv", reader_col="reader_id", text_col="text_id")
```

Expand list-valued interest-area rows:

```python
from brm_eye_workflow.preprocessing import expand_list_valued_ia_table
expand_list_valued_ia_table("data/trial_level.xlsx", "outputs/token_level.csv")
```

Run LLM prompt generation without calling a model:

```bash
python -m brm_eye_workflow.llm_prompts group \
  --input_file selected_with_LoE_examples.xlsx \
  --output_dir outputs/LoE \
  --group_type LoE \
  --dry_run
```

Evaluate an LLM output file:

```python
from brm_eye_workflow.llm_evaluation import evaluate_prediction_file
evaluate_prediction_file(
    "outputs/LoE/output_group_LoE_IA_DWELL_TIME_shot5.xlsx",
    "outputs/eval/LoE_DWL_shot5",
    target_col="IA_DWELL_TIME",
)
```

Run residual calibration:

```python
from brm_eye_workflow.calibration import calibration_curve
curve = calibration_curve(df, y_col="IA_DWELL_TIME", pred_col="pred", reader_col="reader_id")
```

## Repository note

The original CECO corpus should be obtained from its provider under its data-use conditions. This code is intended to be released with split definitions, derived non-identifying outputs, prompt templates, and synthetic demonstration data.
