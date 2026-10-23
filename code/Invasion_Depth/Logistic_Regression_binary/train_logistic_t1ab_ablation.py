"""Train and evaluate logistic regression models for T1a/T1b feature ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_rel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "results_logistic_t1ab_ablation"

SHEET_NAME = "final 随访"
LAYER_COLUMN = "层次"
LAYER_MAP = {"黏膜层": 0, "黏膜下层": 1}
PATIENT_ID_COLUMN = "姓名"
FALLBACK_ID_COLUMN = "编号"
IDENTIFIER_COLUMNS = ("编号", "序号", "姓名", "层次")

N_SPLITS = 5
CV_SEED = 42
MODEL_SEED = 42
PREDICTION_THRESHOLD = 0.5
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 2026
BOOTSTRAP_FOLD = 1
ALPHA = 0.05

LOGISTIC_PARAMETERS: dict[str, object] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": MODEL_SEED,
    "C": 0.1,
    "solver": "lbfgs",
}

MARKER_THRESHOLDS = {
    "PGI": {"value": 70, "direction": "low", "label": "PGI_bin"},
    "PGII": {"value": 21, "direction": "high", "label": "PGII_bin"},
    "PGR": {"value": 7.5, "direction": "low", "label": "PGR_bin"},
    "CEA": {"value": 5, "direction": "high", "label": "CEA_bin"},
    "CA199": {"value": 37, "direction": "high", "label": "CA199_bin"},
    "CA125": {"value": 35, "direction": "high", "label": "CA125_bin"},
    "CA153": {"value": 25, "direction": "high", "label": "CA153_bin"},
    "AFP": {"value": 8.78, "direction": "high", "label": "AFP_bin"},
}

DIFFERENTIATION_MERGE_MAP = {"未分化型": "未分化", "分化型": "分化"}


@dataclass(frozen=True)
class ExperimentSpec:
    """Define one model configuration."""

    key: str
    display: str
    columns: list[str]
    kind: str
    removed_feature: str | None = None


@dataclass
class ExperimentResult:
    """Store cross-validation outputs for one configuration."""

    spec: ExperimentSpec
    oof_probabilities: np.ndarray
    fold_aucs: list[float]
    fold_average_precisions: list[float]

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def std_auc(self) -> float:
        return float(np.std(self.fold_aucs))

    @property
    def mean_average_precision(self) -> float:
        return float(np.mean(self.fold_average_precisions))

    @property
    def std_average_precision(self) -> float:
        return float(np.std(self.fold_average_precisions))


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a source column to numeric values."""

    def parse_value(value: object) -> float:
        text = str(value).strip()
        if text in {".", "nan", "", "None"}:
            return np.nan
        text = (
            text.replace("＜", "")
            .replace("<", "")
            .replace(">", "")
            .replace("≤", "")
            .replace("≥", "")
        )
        try:
            return float(text)
        except ValueError:
            return np.nan

    return series.apply(parse_value)


def clean_binary(series: pd.Series) -> pd.Series:
    """Convert a source column to binary values."""

    def parse_value(value: object) -> float:
        text = str(value).strip()
        if text in {"1", "1.0"}:
            return 1
        if text in {"0", "0.0"}:
            return 0
        return np.nan

    return series.apply(parse_value)


def clean_categorical(series: pd.Series) -> pd.Series:
    """Replace missing categorical values with a stable label."""

    return series.astype(str).replace([".", "nan", "None", ""], "Unknown")


def clean_differentiation(series: pd.Series) -> pd.Series:
    """Normalize differentiation labels from the source workbook."""

    return clean_categorical(series).replace(DIFFERENTIATION_MERGE_MAP)


def binarize_marker(
    series: pd.Series,
    threshold: float,
    direction: str,
) -> pd.Series:
    """Binarize one serum marker at its predefined threshold."""

    numeric = clean_numeric(series)

    def binarize_value(value: float) -> float:
        if np.isnan(value):
            return np.nan
        if direction == "low":
            return 1 if value <= threshold else 0
        return 1 if value > threshold else 0

    return numeric.apply(binarize_value)


def bin_tumor_size(series: pd.Series) -> pd.Series:
    """Encode tumor size as <=2 cm, >2 to 3 cm, or >3 cm."""

    numeric = clean_numeric(series)

    def bin_value(value: float) -> float:
        if np.isnan(value):
            return np.nan
        if value <= 2:
            return 0
        if value <= 3:
            return 1
        return 2

    return numeric.apply(bin_value)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Check that all required source columns are available."""

    required = {
        LAYER_COLUMN,
        "肿瘤大小提取",
        "年龄",
        "大体溃疡",
        "分化类类型",
        "性别",
        *MARKER_THRESHOLDS.keys(),
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required Excel columns: {missing}")


def prompt_for_data_path() -> Path:
    """Request the input workbook path from the user."""

    entered_path = input("Input your dataset path: ").strip().strip('"')
    if not entered_path:
        raise ValueError("No dataset path was provided.")
    return Path(entered_path).expanduser()


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load records with a pure T1a or T1b invasion-depth label."""

    if not data_path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {data_path}")

    df_all = pd.read_excel(data_path, sheet_name=SHEET_NAME)
    validate_input_columns(df_all)

    mask = df_all[LAYER_COLUMN].isin(LAYER_MAP)
    df = df_all.loc[mask].copy().reset_index(drop=True)
    y = df[LAYER_COLUMN].map(LAYER_MAP).to_numpy(dtype=int)

    print(f"Loaded {len(df_all)} rows from {data_path.name}.")
    print(f"Included {len(df)} T1a/T1b rows: {(y == 0).sum()} T1a and {y.sum()} T1b.")
    return df, y


def select_patient_id_column(df: pd.DataFrame) -> str:
    """Select the patient identifier used for bootstrap resampling."""

    if PATIENT_ID_COLUMN in df.columns:
        selected = PATIENT_ID_COLUMN
    elif FALLBACK_ID_COLUMN in df.columns:
        selected = FALLBACK_ID_COLUMN
    else:
        raise KeyError(
            "Neither the primary nor fallback patient identifier column was found."
        )

    print(
        f"Bootstrap unit: {df[selected].nunique()} unique patient identifiers "
        f"across {len(df)} rows."
    )
    return selected


def build_feature_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the complete feature matrix."""

    features = pd.DataFrame(index=df.index)
    marker_features: list[str] = []

    for source_column, config in MARKER_THRESHOLDS.items():
        output_column = str(config["label"])
        features[output_column] = binarize_marker(
            df[source_column],
            float(config["value"]),
            str(config["direction"]),
        )
        marker_features.append(output_column)

    features["Tumor_Size_Bin"] = bin_tumor_size(df["肿瘤大小提取"])
    features["Age"] = clean_numeric(df["年龄"])
    features["Gross_Ulcer"] = clean_binary(df["大体溃疡"])

    categorical_columns = {
        "分化类类型": "Differentiation",
        "性别": "Sex",
    }
    for source_column, output_column in categorical_columns.items():
        encoder = LabelEncoder()
        if source_column == "分化类类型":
            cleaned = clean_differentiation(df[source_column])
        else:
            cleaned = clean_categorical(df[source_column])
        features[output_column] = encoder.fit_transform(cleaned)

    base_features = [
        "Tumor_Size_Bin",
        "Age",
        "Gross_Ulcer",
        "Differentiation",
        "Sex",
    ]

    print(f"Prepared {len(base_features)} reference features and {len(marker_features)} markers.")
    return features, base_features, marker_features


def build_experiment_registry(
    base_features: list[str],
    marker_features: list[str],
) -> list[ExperimentSpec]:
    """Define the reference, marker-addition, and ablation configurations."""

    experiments = [
        ExperimentSpec(
            key="01_reference",
            display="Reference",
            columns=list(base_features),
            kind="reference",
        ),
        ExperimentSpec(
            key="02_reference_plus_blood_markers",
            display="Reference + blood markers",
            columns=list(base_features) + list(marker_features),
            kind="addition",
        ),
    ]

    config_labels = [
        "Config. 1 - Tumour size",
        "Config. 2 - Age",
        "Config. 3 - Ulceration",
        "Config. 4 - Differentiation",
        "Config. 5 - Sex",
    ]
    for index, (feature, display) in enumerate(
        zip(base_features, config_labels),
        start=3,
    ):
        experiments.append(
            ExperimentSpec(
                key=f"{index:02d}_ablate_{feature}",
                display=display,
                columns=[column for column in base_features if column != feature],
                kind="ablation",
                removed_feature=feature,
            )
        )

    return experiments


def make_logistic_model() -> LogisticRegression:
    """Create the logistic regression model."""

    return LogisticRegression(**LOGISTIC_PARAMETERS)


def run_cross_validation(
    spec: ExperimentSpec,
    x: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> ExperimentResult:
    """Run five-fold cross-validation for one configuration."""

    x_array = x.to_numpy()
    oof_probabilities = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    fold_average_precisions: list[float] = []

    print(f"\nRunning {spec.display} with {len(spec.columns)} features.")
    for fold, (train_index, validation_index) in enumerate(cv.split(x_array, y), start=1):
        x_train = x_array[train_index]
        x_validation = x_array[validation_index]
        y_train = y[train_index]
        y_validation = y[validation_index]

        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x_train)
        x_validation = imputer.transform(x_validation)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)

        model = make_logistic_model()
        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_validation)[:, 1]
        oof_probabilities[validation_index] = probabilities
        fold_auc = float(roc_auc_score(y_validation, probabilities))
        fold_average_precision = float(
            average_precision_score(y_validation, probabilities)
        )
        fold_aucs.append(fold_auc)
        fold_average_precisions.append(fold_average_precision)

        print(
            f"  Fold {fold}: AUROC={fold_auc:.4f}, "
            f"AP={fold_average_precision:.4f}."
        )

    return ExperimentResult(
        spec=spec,
        oof_probabilities=oof_probabilities,
        fold_aucs=fold_aucs,
        fold_average_precisions=fold_average_precisions,
    )


def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calculate classification metrics at the fixed threshold."""

    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    report = classification_report(
        y_true,
        predictions,
        target_names=["T1a (0)", "T1b (1)"],
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, object] = {
        "threshold": PREDICTION_THRESHOLD,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    return predictions, metrics


def run_experiment(
    spec: ExperimentSpec,
    all_features: pd.DataFrame,
    y: np.ndarray,
    source_data: pd.DataFrame,
    cv: StratifiedKFold,
    output_root: Path,
) -> ExperimentResult:
    """Train one configuration and save its numeric outputs."""

    output_directory = output_root / spec.key
    output_directory.mkdir(parents=True, exist_ok=True)

    result = run_cross_validation(
        spec,
        all_features[spec.columns].copy(),
        y,
        cv,
    )

    oof_auc = float(roc_auc_score(y, result.oof_probabilities))
    oof_average_precision = float(
        average_precision_score(y, result.oof_probabilities)
    )
    predictions, threshold_metrics = calculate_threshold_metrics(
        y,
        result.oof_probabilities,
    )

    identifier_columns = [
        column for column in IDENTIFIER_COLUMNS if column in source_data.columns
    ]
    prediction_table = source_data[identifier_columns].copy().reset_index(drop=True)
    prediction_table["predicted_probability"] = result.oof_probabilities
    prediction_table["predicted_label"] = predictions
    prediction_table["correct"] = (predictions == y).astype(int)
    prediction_table.to_csv(
        output_directory / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "fold": range(1, len(result.fold_aucs) + 1),
            "auc": result.fold_aucs,
            "average_precision": result.fold_average_precisions,
        }
    ).to_csv(
        output_directory / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "key": spec.key,
        "display": spec.display,
        "kind": spec.kind,
        "removed_feature": spec.removed_feature,
        "n_features": len(spec.columns),
        "features": spec.columns,
        "model_parameters": LOGISTIC_PARAMETERS,
        "mean_fold_auc": result.mean_auc,
        "std_fold_auc": result.std_auc,
        "mean_fold_average_precision": result.mean_average_precision,
        "std_fold_average_precision": result.std_average_precision,
        "oof_auc": oof_auc,
        "oof_average_precision": oof_average_precision,
        "fold_aucs": result.fold_aucs,
        "fold_average_precisions": result.fold_average_precisions,
        **threshold_metrics,
    }
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        f"  Mean AUROC={result.mean_auc:.4f}; "
        f"mean AP={result.mean_average_precision:.4f}; "
        f"OOF AUROC={oof_auc:.4f}; OOF AP={oof_average_precision:.4f}."
    )
    print(
        f"  Threshold {PREDICTION_THRESHOLD:.1f}: "
        f"accuracy={threshold_metrics['accuracy']:.4f}, "
        f"balanced_accuracy={threshold_metrics['balanced_accuracy']:.4f}, "
        f"sensitivity={threshold_metrics['sensitivity']:.4f}, "
        f"specificity={threshold_metrics['specificity']:.4f}."
    )
    return result


def build_summary_table(
    results: list[ExperimentResult],
    y: np.ndarray,
) -> pd.DataFrame:
    """Create the cross-validation summary table."""

    reference = results[0]
    rows = []
    for result in results:
        rows.append(
            {
                "key": result.spec.key,
                "experiment": result.spec.display,
                "kind": result.spec.kind,
                "removed_feature": result.spec.removed_feature or "",
                "n_features": len(result.spec.columns),
                "mean_fold_auc": result.mean_auc,
                "std_fold_auc": result.std_auc,
                "mean_fold_average_precision": result.mean_average_precision,
                "std_fold_average_precision": result.std_average_precision,
                "oof_auc": float(roc_auc_score(y, result.oof_probabilities)),
                "oof_average_precision": float(
                    average_precision_score(y, result.oof_probabilities)
                ),
                "mean_auc_difference_from_reference": (
                    result.mean_auc - reference.mean_auc
                ),
            }
        )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Apply the Benjamini-Hochberg false-discovery-rate correction."""

    values = np.asarray(p_values, dtype=float)
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=float)
    running_minimum = 1.0

    for rank in range(count - 1, -1, -1):
        index = order[rank]
        candidate = values[index] * count / (rank + 1)
        running_minimum = min(running_minimum, candidate)
        adjusted[index] = running_minimum

    return adjusted


def run_paired_fold_tests(results: list[ExperimentResult]) -> pd.DataFrame:
    """Compare each configuration with the reference across matched folds."""

    reference = results[0]
    rows = []

    for result in results[1:]:
        reference_aucs = np.asarray(reference.fold_aucs)
        comparison_aucs = np.asarray(result.fold_aucs)
        differences = reference_aucs - comparison_aucs

        if np.allclose(differences, 0):
            t_statistic, p_value = 0.0, 1.0
        else:
            t_statistic, p_value = ttest_rel(reference_aucs, comparison_aucs)

        rows.append(
            {
                "key": result.spec.key,
                "experiment": result.spec.display,
                "removed_feature": result.spec.removed_feature or "",
                "mean_auc_difference_reference_minus_comparator": float(
                    np.mean(differences)
                ),
                "sd_auc_difference": float(np.std(differences, ddof=1)),
                "paired_t_statistic": float(t_statistic),
                "paired_t_p_value": float(p_value),
            }
        )

    table = pd.DataFrame(rows)
    table["paired_t_bh_adjusted_p_value"] = benjamini_hochberg(
        table["paired_t_p_value"].to_numpy()
    )
    table["paired_t_significant_bh_0_05"] = (
        table["paired_t_bh_adjusted_p_value"] < ALPHA
    )
    return table


def build_patient_groups(
    validation_indices: np.ndarray,
    patient_ids: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[Hashable, list[int]], dict[Hashable, int], np.ndarray]:
    """Group validation rows by patient identifier."""

    patient_to_rows: dict[Hashable, list[int]] = {}
    patient_to_class: dict[Hashable, int] = {}

    for row_index in validation_indices:
        patient_id = patient_ids[row_index]
        patient_to_rows.setdefault(patient_id, []).append(int(row_index))
        patient_to_class[patient_id] = int(y[row_index])

    patient_array = np.asarray(list(patient_to_rows), dtype=object)
    return patient_to_rows, patient_to_class, patient_array


def generate_bootstrap_indices(
    patient_to_rows: dict[Hashable, list[int]],
    patient_to_class: dict[Hashable, int],
    patient_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
    max_retry: int = 200,
) -> list[np.ndarray]:
    """Generate patient-level bootstrap samples."""

    random_generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    rejected = 0

    while len(samples) < n_bootstrap:
        sampled_patients = random_generator.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )
        if len({patient_to_class[patient] for patient in sampled_patients}) < 2:
            rejected += 1
            if rejected > max_retry * n_bootstrap:
                raise RuntimeError(
                    "Too many single-class bootstrap samples. Check the validation data."
                )
            continue

        sampled_rows: list[int] = []
        for patient in sampled_patients:
            sampled_rows.extend(patient_to_rows[patient])
        samples.append(np.asarray(sampled_rows, dtype=int))

    return samples


def bootstrap_auc_distribution(
    probabilities: np.ndarray,
    bootstrap_indices: list[np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """Calculate AUROC for each bootstrap sample."""

    distribution = np.empty(len(bootstrap_indices), dtype=float)
    for index, rows in enumerate(bootstrap_indices):
        distribution[index] = roc_auc_score(y[rows], probabilities[rows])
    return distribution


def run_bootstrap_analysis(
    results: list[ExperimentResult],
    y: np.ndarray,
    source_data: pd.DataFrame,
    cv: StratifiedKFold,
    patient_id_column: str,
) -> pd.DataFrame:
    """Run the fixed-fold patient-level bootstrap analysis."""

    split_indices = [validation for _, validation in cv.split(np.zeros((len(y), 1)), y)]
    validation_indices = split_indices[BOOTSTRAP_FOLD - 1]
    patient_to_rows, patient_to_class, patient_ids = build_patient_groups(
        validation_indices,
        source_data[patient_id_column].to_numpy(),
        y,
    )

    print(
        f"\nRunning {N_BOOTSTRAP} bootstrap samples on fold {BOOTSTRAP_FOLD} "
        f"for {len(results)} configurations."
    )

    bootstrap_aucs: dict[str, np.ndarray] = {}
    observed_aucs: dict[str, float] = {}
    for offset, result in enumerate(results):
        bootstrap_indices = generate_bootstrap_indices(
            patient_to_rows,
            patient_to_class,
            patient_ids,
            N_BOOTSTRAP,
            BOOTSTRAP_SEED + offset,
        )
        bootstrap_aucs[result.spec.key] = bootstrap_auc_distribution(
            result.oof_probabilities,
            bootstrap_indices,
            y,
        )
        observed_aucs[result.spec.key] = float(
            roc_auc_score(
                y[validation_indices],
                result.oof_probabilities[validation_indices],
            )
        )

    reference = results[0]
    rows = []
    for result in results[1:]:
        delta_auc = (
            bootstrap_aucs[result.spec.key] - bootstrap_aucs[reference.spec.key]
        )
        null_distribution = np.zeros_like(delta_auc)

        if np.array_equal(delta_auc, null_distribution):
            u_statistic = float(len(delta_auc) ** 2) / 2.0
            p_value = 1.0
        else:
            u_statistic, p_value = mannwhitneyu(
                delta_auc,
                null_distribution,
                alternative="two-sided",
                method="asymptotic",
            )

        rows.append(
            {
                "key": result.spec.key,
                "comparison": f"{result.spec.display} vs {reference.spec.display}",
                "removed_feature": result.spec.removed_feature or "",
                "observed_delta_auc_comparator_minus_reference": (
                    observed_aucs[result.spec.key]
                    - observed_aucs[reference.spec.key]
                ),
                "median_bootstrap_delta_auc_comparator_minus_reference": float(
                    np.median(delta_auc)
                ),
                "mann_whitney_u": float(u_statistic),
                "raw_p_value": float(p_value),
            }
        )

    table = pd.DataFrame(rows)
    table["bh_adjusted_p_value"] = benjamini_hochberg(
        table["raw_p_value"].to_numpy()
    )
    table["significant_bh_0_05"] = table["bh_adjusted_p_value"] < ALPHA
    return table


def main() -> None:
    """Run all analyses and save numeric outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = prompt_for_data_path()
    print("Logistic regression T1a/T1b feature-ablation analysis")
    print(f"Input workbook: {data_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    source_data, y = load_dataset(data_path)
    patient_id_column = select_patient_id_column(source_data)
    all_features, base_features, marker_features = build_feature_frame(source_data)
    experiments = build_experiment_registry(base_features, marker_features)

    print(f"Running {len(experiments)} model configurations.")
    print(
        "Model: LogisticRegression(C=0.1, solver='lbfgs', "
        "class_weight='balanced', max_iter=1000)."
    )
    print("Fold pipeline: median imputation, standardization, logistic regression.")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_SEED,
    )
    results = [
        run_experiment(
            experiment,
            all_features,
            y,
            source_data,
            cv,
            OUTPUT_DIR,
        )
        for experiment in experiments
    ]

    paired_tests = run_paired_fold_tests(results)
    paired_tests.to_csv(
        OUTPUT_DIR / "paired_fold_ttest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary_table(results, y)
    summary = summary.merge(
        paired_tests[
            [
                "key",
                "mean_auc_difference_reference_minus_comparator",
                "paired_t_p_value",
                "paired_t_bh_adjusted_p_value",
                "paired_t_significant_bh_0_05",
            ]
        ],
        on="key",
        how="left",
    )
    summary.to_csv(
        OUTPUT_DIR / "all_experiment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_results = run_bootstrap_analysis(
        results,
        y,
        source_data,
        cv,
        patient_id_column,
    )
    bootstrap_results.to_csv(
        OUTPUT_DIR / "bootstrap_mannwhitney_vs_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAnalysis complete.")
    print(f"Results were saved to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
