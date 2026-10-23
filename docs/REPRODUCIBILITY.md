# Reproducibility notes

This repository provides the predictive-modelling framework described in the associated manuscript. Reproducing the manuscript results requires the controlled study datasets, the original image collection, compatible model weights or API endpoints, and the software versions listed in `requirements.txt`.

## Dataset roles

The experiments use a controlled internal development dataset. Logistic-regression and LightGBM models additionally use a separate external dataset for validation. Task-specific deep-learning and VLM experiments are evaluated on their prespecified internal image-available analysis sets and do not use the external validation dataset. The study workbooks and images are not distributed in this repository, and the public scripts request their paths at runtime.

## Analysis coverage

| Manuscript component | Repository coverage |
|---|---|
| Figure 5 and related extended-data analyses | Logistic-regression and LightGBM training, feature ablation, bootstrap comparison, internal evaluation, and external validation |
| Figure 6 and related extended-data analyses | Task-specific deep-learning and VLM benchmarking across image-only, clinical-text-only or tabular-only, and multimodal inputs |
| Figure 7 | Deep-learning checkpoints, predictions, and attention-weight outputs needed for local case review; manuscript figure assembly is not included |

## Evaluation design retained in the code

- Five-fold stratified cross-validation for internal modelling.
- Separate internal and external datasets for traditional-model validation.
- Continuous probabilities for AUROC and average precision.
- Argmax classification for the three-class invasion-depth task.
- Prespecified feature-ablation configurations.
- Patient-level bootstrap comparisons where used in the manuscript.

## Task-to-code map

### Lymph node metastasis

| Model family | Training or internal analysis | External validation |
|---|---|---|
| Logistic regression | `code/Lymph_Node_Metastasis/Logistic_Regression/train_logistic_lnm_ablation.py` | `code/Lymph_Node_Metastasis/Logistic_Regression/validate_logistic_lnm_external.py` |
| LightGBM | `code/Lymph_Node_Metastasis/LightGBM/train_lightgbm_lnm_ablation.py` | `code/Lymph_Node_Metastasis/LightGBM/validate_lightgbm_lnm_external.py` |
| Task-specific deep learning | `code/Lymph_Node_Metastasis/Deep_learning/train_deep_learning_lnm.py` | — |
| Vision-language models | `code/Lymph_Node_Metastasis/VLM_*/` | — |

### Invasion depth

| Model family | Binary task | Three-class task |
|---|---|---|
| Logistic regression | `code/Invasion_Depth/Logistic_Regression_binary/` | `code/Invasion_Depth/Logistic_Regression_multiclass/` |
| LightGBM | `code/Invasion_Depth/LightGBM_binary/` | `code/Invasion_Depth/LightGBM_multiclass/` |
| Task-specific deep learning | `code/Invasion_Depth/Deep_learning_binary/train_deep_learning_invasion_binary.py` | `code/Invasion_Depth/Deep_learning_multiclass/train_deep_learning_invasion_multiclass.py` |
| Vision-language models | `code/Invasion_Depth/VLM_*/` | — |

## Reproduction boundaries

The study data and images are not public, so an independent user cannot reproduce the manuscript results from this repository alone. Researchers can inspect the full modelling logic, adapt the framework to an appropriately governed dataset, and compare their own results using the same task definitions and metrics.

Cloud-hosted VLM results also depend on provider-side model versions, service availability, access permissions, and endpoint configuration. The scripts retain the model identifiers used in the study, but future provider changes may affect reproducibility.

## Release checks

Before publishing a new revision:

1. run `python -m compileall -q code`;
2. search for local absolute paths, dataset filenames, credentials, tokens, and private endpoint identifiers;
3. confirm that no spreadsheet, image, prediction, checkpoint, or `results_*` directory is staged;
4. verify the README, citation metadata, manuscript title, author list, and data-sharing statement against the current manuscript;
5. run end-to-end analyses only in the approved private environment and compare outputs with the manuscript tables and figures.
