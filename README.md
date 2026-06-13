# Leakage-Aware Eye-Movement Prediction Workflow

This repository contains the reproducible code for the paper:

**A Leakage-Aware Workflow for Evaluating L2 Eye-Movement Prediction Under Participant and Item Shifts**

The code implements the main analysis workflow used in the manuscript: interest-area data preprocessing, participant/item split construction, split-aware lexical surprisal, supervised baseline evaluation, metadata-leakage diagnostics, small-sample reader calibration, LLM prompt generation, and LLM output evaluation.

## 1. Repository structure

```text
brm_eye_workflow/
  __init__.py
  constants.py              # shared target names and label mappings
  io_utils.py               # table I/O and safe parsing helpers
  preprocessing.py          # IA table reshaping, missing values, background summaries, lexical-feature merging
  splits.py                 # TEXT, READER, and RXT domain-separated splits
  surprisal.py              # split-aware n-gram surprisal
  modeling.py               # ElasticNet and Poisson supervised baselines
  metadata_diagnostics.py   # reader-level metadata permutation and CKA diagnostics
  calibration.py            # residual small-sample reader calibration
  llm_prompts.py            # group/individual LLM prompt construction and local Ollama inference
  llm_evaluation.py         # LLM output parsing, token-level metrics, and group-level diagnostics
requirements.txt
README.md
```

## 2. Installation

Create a clean Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include `pandas`, `numpy`, `scikit-learn`, `scipy`, `requests`, `openpyxl`, and `tqdm`.

The LLM prompt module can call a local Ollama-compatible endpoint. This is optional. Use `--dry_run` to generate prompts without calling a model.

## 3. Expected data format

The code is designed for tabular interest-area reading data. The original corpus is not included in this repository. Researchers should prepare a non-identifying table with columns equivalent to the following:

| Column                      | Meaning                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `reader_id`                 | participant or reader identifier                             |
| `text_id`                   | passage, sentence, or trial/text identifier                  |
| `IA_LABEL`                  | token or interest-area label; either one token per row or a list-valued trial-level cell |
| eye-movement target columns | e.g., `IA_FIRST_FIXATION_DURATION`, `IA_DWELL_TIME`, `IA_SKIP` |
| lexical feature columns     | e.g., word length, frequency, surprisal, HSK level, or other word-level variables |
| reader metadata columns     | e.g., country group, age of acquisition, length of exposure, proficiency |

The default target names are defined in `brm_eye_workflow/constants.py`:

```python
from brm_eye_workflow.constants import EYE_TARGETS
print(EYE_TARGETS)
```

## 4. Workflow overview

The workflow has six main stages.

### Stage 1: Preprocess interest-area tables

Use `preprocessing.py` to reshape data between trial-level list-valued rows and token-level rows.

```python
from brm_eye_workflow.preprocessing import expand_list_valued_ia_table

expand_list_valued_ia_table(
    input_path="data/trial_level.xlsx",
    output_path="outputs/token_level.csv",
    id_cols=("reader_id", "text_id"),
    list_cols=("IA_LABEL", "IA_DWELL_TIME", "IA_SKIP"),
)
```

To combine token-level rows back into trial-level list-valued rows:

```python
from brm_eye_workflow.preprocessing import combine_interest_area_rows

combine_interest_area_rows(
    input_path="data/token_level.csv",
    output_path="outputs/trial_level.csv",
    group_cols=("reader_id", "text_id"),
)
```

To summarize participant background variables:

```python
from brm_eye_workflow.preprocessing import summarize_background

summarize_background(
    background_path="data/background.csv",
    group_col="country",
    numeric_cols=("age", "HSK", "AoA", "LoE"),
    output_path="outputs/background_summary.csv",
)
```

### Stage 2: Construct domain-separated splits

Use `splits.py` to create the three split types used in the manuscript:

- `split_TEXT`: held-out texts/items, with readers allowed to overlap;
- `split_READER`: held-out readers, with texts allowed to overlap;
- `split_RXT`: joint reader-text held-out split.

```python
from brm_eye_workflow.splits import make_domain_splits, summarize_split_counts

splits = make_domain_splits(
    input_path="outputs/token_level.csv",
    output_path="outputs/token_level_with_splits.csv",
    reader_col="reader_id",
    text_col="text_id",
    seed=13,
)

split_counts = summarize_split_counts(splits)
split_counts.to_csv("outputs/split_counts.csv", index=False)
```

Rows marked as `unused` in `split_RXT` are not part of the strict joint reader-text split.

### Stage 3: Add split-aware surprisal

Use `surprisal.py` to estimate n-gram surprisal from training texts only. This prevents held-out text information from entering the lexical feature pipeline.

```python
from brm_eye_workflow.io_utils import read_table, write_table
from brm_eye_workflow.surprisal import add_split_aware_surprisal

trial_df = read_table("outputs/trial_level_with_splits.csv")
trial_df = add_split_aware_surprisal(
    trial_df,
    token_list_col="IA_LABEL",
    split_col="split_TEXT",
    train_label="train",
    n=3,
    alpha=0.1,
    output_col="surprisal",
)
write_table(trial_df, "outputs/trial_level_with_surprisal.csv")
```

### Stage 4: Fit supervised lexical baselines

Use `modeling.py` for simple supervised baselines. The module provides ElasticNet and Poisson regression helpers.

```python
from brm_eye_workflow.io_utils import read_table
from brm_eye_workflow.modeling import fit_evaluate_by_split
from brm_eye_workflow.constants import EYE_TARGETS

DF = read_table("outputs/token_level_with_splits.csv")
feature_cols = ["length", "log_frequency", "surprisal", "hsk"]

results = fit_evaluate_by_split(
    DF,
    feature_cols=feature_cols,
    target_cols=EYE_TARGETS,
    split_col="split_READER",
    model_name="elasticnet",
)
results.to_csv("outputs/supervised_reader_results.csv", index=False)
```

For nonnegative count-like targets, use `model_name="poisson"`.

### Stage 5: Diagnose metadata leakage

Use `metadata_diagnostics.py` to test whether learner-background variables provide transferable information or act as reader-indexing cues.

Reader-level permutation:

```python
from brm_eye_workflow.metadata_diagnostics import reader_level_permute

metadata_cols = ["age", "AoA", "LoE", "HSK"]
DF_perm = reader_level_permute(
    DF,
    reader_col="reader_id",
    metadata_cols=metadata_cols,
    seed=13,
)
```

CKA diagnostic between metadata and reader identity:

```python
from brm_eye_workflow.metadata_diagnostics import cka_metadata_identity_test

cka = cka_metadata_identity_test(
    reader_metadata=DF[["reader_id", *metadata_cols]].drop_duplicates(),
    reader_col="reader_id",
    metadata_cols=metadata_cols,
    n_perm=500,
    seed=13,
)
print(cka)
```

Feature-gain comparison:

```python
from brm_eye_workflow.metadata_diagnostics import compute_feature_gain_table

gains = compute_feature_gain_table(
    results=all_results,
    baseline_config="word",
    all_config="all",
    perm_config="all-permute",
    metric_col="R2",
    keys=("split", "model", "target"),
)
gains.to_csv("outputs/metadata_gain_table.csv", index=False)
```

Here, `all_results` should contain one row per split/model/target/config combination.

### Stage 6: Estimate small-sample reader calibration effects

Use `calibration.py` to estimate how much target-reader data are needed to improve predictions for unseen readers.

```python
from brm_eye_workflow.calibration import calibration_curve

curve = calibration_curve(
    DF,
    y_col="IA_DWELL_TIME",
    pred_col="pred",
    reader_col="reader_id",
    ratios=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1),
    repeats=20,
    shrinkage=10.0,
    seed=13,
)
curve.to_csv("outputs/calibration_curve_DWL.csv", index=False)
```

`pred_col` should contain predictions from a baseline model fit without using the calibration rows.

## 5. LLM prompt generation and evaluation

### Generate prompts without model calls

Use `--dry_run` to inspect prompts and save them to output files.

```bash
python -m brm_eye_workflow.llm_prompts group \
  --input_file data/selected_with_LoE_examples.xlsx \
  --output_dir outputs/llm_prompts/LoE \
  --group_type LoE \
  --max_shot 5 \
  --dry_run
```

For individual-level prompting:

```bash
python -m brm_eye_workflow.llm_prompts individual \
  --input_file data/selected_with_individual_examples.xlsx \
  --output_dir outputs/llm_prompts/individual \
  --max_shot 5 \
  --dry_run
```

### Run local Ollama inference

Start an Ollama-compatible server first, then remove `--dry_run`:

```bash
python -m brm_eye_workflow.llm_prompts group \
  --input_file data/selected_with_LoE_examples.xlsx \
  --output_dir outputs/llm/LoE \
  --group_type LoE \
  --model_name llama3.3:70b \
  --max_shot 5
```

### Evaluate LLM outputs

Use `llm_evaluation.py` to parse generated numeric lists and compute token-level and group-level diagnostics.

```python
from brm_eye_workflow.llm_evaluation import evaluate_prediction_file

evaluate_prediction_file(
    input_file="outputs/llm/LoE/output_group_LoE_IA_DWELL_TIME_shot5.xlsx",
    output_prefix="outputs/eval/LoE_DWL_shot5",
    target_col="IA_DWELL_TIME",
    output_col="OUTPUT",
    token_col="IA_LABEL",
    group_cols=("country", "AoA_bin", "LoE_bin", "HSK_bin"),
)
```

The evaluator writes four files:

```text
*.tokens.csv          # token-level aligned records
*.errors.csv          # parsing or length-mismatch logs
*.token_metrics.csv   # MAE, RMSE, R2, Pearson, Spearman, Kendall, Wasserstein
*.group_metrics.csv   # group-order recovery and worst-group absolute bias
```

## 6. Minimal end-to-end example

```python
from brm_eye_workflow.constants import EYE_TARGETS
from brm_eye_workflow.io_utils import read_table
from brm_eye_workflow.splits import make_domain_splits
from brm_eye_workflow.modeling import fit_evaluate_by_split
from brm_eye_workflow.metadata_diagnostics import cka_metadata_identity_test

# 1. Add domain-separated split labels.
df = make_domain_splits(
    "data/token_level.csv",
    "outputs/token_level_with_splits.csv",
    reader_col="reader_id",
    text_col="text_id",
)

# 2. Fit a lexical baseline under reader-disjoint evaluation.
feature_cols = ["length", "log_frequency", "surprisal", "hsk"]
results = fit_evaluate_by_split(
    df,
    feature_cols=feature_cols,
    target_cols=EYE_TARGETS,
    split_col="split_READER",
    model_name="elasticnet",
)
results.to_csv("outputs/reader_split_elasticnet.csv", index=False)

# 3. Diagnose whether metadata is coupled with reader identity.
metadata_cols = ["age", "AoA", "LoE", "HSK"]
cka = cka_metadata_identity_test(
    reader_metadata=df[["reader_id", *metadata_cols]].drop_duplicates(),
    reader_col="reader_id",
    metadata_cols=metadata_cols,
)
print(cka)
```
