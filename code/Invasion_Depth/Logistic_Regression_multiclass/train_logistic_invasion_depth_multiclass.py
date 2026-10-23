"""Train and evaluate logistic regression for three-class invasion depth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results_logistic_invasion_depth_multiclass"
)

SHEET_NAME = "final 随访"
SM_COLUMN = "SM"
SM_COLUMN_INDEX = 102
SM_CLASS_MAP = {".": 0, "1": 1, "2": 2, "3": 2}
CLASS_NAMES = ("M", "SM1", "SM2&3")
N_CLASSES = len(CLASS_NAMES)

FEATURE_ORDER = (
    "Tumor_Size_Bin",
    "Age",
    "Gross_Ulcer",
    "Differentiation",
    "Sex",
)
IDENTIFIER_COLUMNS = ("编号", "序号", "姓名", "SM_raw", "层次")
DIFFERENTIATION_MERGE_MAP = {"未分化型": "未分化", "分化型": "分化"}

N_SPLITS = 5
CV_SEED = 42
MODEL_SEED = 42
MEAN_ROC_POINTS = 500

LOGISTIC_PARAMETERS: dict[str, object] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": MODEL_SEED,
    "C": 0.1,
    "solver": "lbfgs",
}


@dataclass
class CrossValidationResult:
    """Store out-of-fold predictions and fold-level metrics."""

    oof_probabilities: np.ndarray
    fold_assignments: np.ndarray
    fold_macro_aucs: list[float]
    fold_macro_average_precisions: list[float]
    fold_class_aucs: dict[int, list[float]]

    @property
    def mean_macro_auc(self) -> float:
        return float(np.mean(self.fold_macro_aucs))

    @property
    def std_macro_auc(self) -> float:
        return float(np.std(self.fold_macro_aucs))

    @property
    def mean_macro_average_precision(self) -> float:
        return float(np.mean(self.fold_macro_average_precisions))

    @property
    def std_macro_average_precision(self) -> float:
        return float(np.std(self.fold_macro_average_precisions))


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


def validate_input_schema(df: pd.DataFrame) -> None:
    """Check the worksheet layout and required source columns."""

    if len(df.columns) <= SM_COLUMN_INDEX:
        raise ValueError(
            f"The worksheet must contain at least {SM_COLUMN_INDEX + 1} columns."
        )
    if df.columns[SM_COLUMN_INDEX] != SM_COLUMN:
        raise ValueError(
            f"Expected column {SM_COLUMN!r} at zero-based position "
            f"{SM_COLUMN_INDEX}, but found {df.columns[SM_COLUMN_INDEX]!r}."
        )

    required = {
        "肿瘤大小提取",
        "年龄",
        "大体溃疡",
        "分化类类型",
        "性别",
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
    """Load records with an M, SM1, SM2, or SM3 pathological label."""

    if not data_path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {data_path}")

    workbook = pd.ExcelFile(data_path)
    if SHEET_NAME not in workbook.sheet_names:
        raise KeyError(f"Missing required Excel worksheet: {SHEET_NAME!r}")

    df_all = workbook.parse(SHEET_NAME)
    validate_input_schema(df_all)

    sm_raw = df_all.iloc[:, SM_COLUMN_INDEX].astype(str).str.strip()
    target = sm_raw.map(SM_CLASS_MAP)
    included = target.notna()

    df = df_all.loc[included].copy().reset_index(drop=True)
    df["SM_raw"] = sm_raw.loc[included].to_numpy()
    y = target.loc[included].to_numpy(dtype=int)

    class_counts = np.bincount(y, minlength=N_CLASSES)
    print(f"Loaded {len(df_all)} rows from {data_path.name}.")
    print(f"Included {len(df)} rows with valid three-class invasion-depth labels.")
    print(
        "Class counts: "
        + ", ".join(
            f"{class_name}={int(class_counts[index])}"
            for index, class_name in enumerate(CLASS_NAMES)
        )
        + "."
    )
    return df, y


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the five-feature reference model matrix."""

    features = pd.DataFrame(index=df.index)
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

    missing_rates = features[list(FEATURE_ORDER)].isna().mean()
    missing_features = missing_rates[missing_rates > 0]
    if not missing_features.empty:
        details = ", ".join(
            f"{feature}={rate:.1%}"
            for feature, rate in missing_features.items()
        )
        print(f"Missing feature values: {details}.")

    print(f"Prepared the {len(FEATURE_ORDER)} reference-model features.")
    return features[list(FEATURE_ORDER)]


def make_logistic_model() -> LogisticRegression:
    """Create the multinomial logistic regression model."""

    return LogisticRegression(**LOGISTIC_PARAMETERS)


def multiclass_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Calculate macro one-vs-rest AUROC."""

    return float(
        roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    )


def multiclass_average_precision(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Calculate macro one-vs-rest average precision."""

    binary_targets = label_binarize(y_true, classes=range(N_CLASSES))
    return float(
        average_precision_score(
            binary_targets,
            probabilities,
            average="macro",
        )
    )


def run_cross_validation(
    features: pd.DataFrame,
    y: np.ndarray,
) -> CrossValidationResult:
    """Run the fold-specific preprocessing and five-fold evaluation."""

    x_array = features.to_numpy()
    oof_probabilities = np.zeros((len(y), N_CLASSES), dtype=float)
    fold_assignments = np.zeros(len(y), dtype=int)
    fold_macro_aucs: list[float] = []
    fold_macro_average_precisions: list[float] = []
    fold_class_aucs = {index: [] for index in range(N_CLASSES)}

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_SEED,
    )

    print("Running five-fold cross-validation.")
    for fold, (train_index, validation_index) in enumerate(
        cv.split(x_array, y),
        start=1,
    ):
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

        probabilities = model.predict_proba(x_validation)
        oof_probabilities[validation_index] = probabilities
        fold_assignments[validation_index] = fold

        fold_auc = multiclass_auc(y_validation, probabilities)
        fold_average_precision = multiclass_average_precision(
            y_validation,
            probabilities,
        )
        fold_macro_aucs.append(fold_auc)
        fold_macro_average_precisions.append(fold_average_precision)

        for class_index in range(N_CLASSES):
            fold_class_aucs[class_index].append(
                float(
                    roc_auc_score(
                        (y_validation == class_index).astype(int),
                        probabilities[:, class_index],
                    )
                )
            )

        print(
            f"  Fold {fold}: macro AUROC={fold_auc:.4f}, "
            f"macro AP={fold_average_precision:.4f}"
        )

    return CrossValidationResult(
        oof_probabilities=oof_probabilities,
        fold_assignments=fold_assignments,
        fold_macro_aucs=fold_macro_aucs,
        fold_macro_average_precisions=fold_macro_average_precisions,
        fold_class_aucs=fold_class_aucs,
    )


def calculate_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calculate classification metrics using the argmax decision rule."""

    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=range(N_CLASSES),
    )
    report = classification_report(
        y_true,
        predictions,
        labels=range(N_CLASSES),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, object] = {
        "decision_rule": "argmax",
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    return predictions, metrics


def build_fold_metrics(result: CrossValidationResult) -> pd.DataFrame:
    """Create the fold-level metric table."""

    table = pd.DataFrame(
        {
            "fold": range(1, N_SPLITS + 1),
            "macro_auc_ovr": result.fold_macro_aucs,
            "macro_average_precision": result.fold_macro_average_precisions,
        }
    )
    for class_index, class_name in enumerate(CLASS_NAMES):
        table[f"auc_{class_name}_vs_rest"] = result.fold_class_aucs[
            class_index
        ]
    return table


def build_mean_roc_table(
    y: np.ndarray,
    result: CrossValidationResult,
) -> pd.DataFrame:
    """Calculate class-specific mean ROC profiles across validation folds."""

    common_fpr = np.linspace(0, 1, MEAN_ROC_POINTS)
    table = pd.DataFrame({"false_positive_rate": common_fpr})

    for class_index, class_name in enumerate(CLASS_NAMES):
        interpolated_tprs = []
        for fold in range(1, N_SPLITS + 1):
            fold_mask = result.fold_assignments == fold
            binary_target = (y[fold_mask] == class_index).astype(int)
            fpr, tpr, _ = roc_curve(
                binary_target,
                result.oof_probabilities[fold_mask, class_index],
            )
            interpolation = interp1d(
                fpr,
                tpr,
                kind="linear",
                bounds_error=False,
                fill_value=(0.0, 1.0),
            )
            interpolated_tprs.append(interpolation(common_fpr))

        tpr_array = np.asarray(interpolated_tprs)
        table[f"mean_true_positive_rate_{class_name}"] = tpr_array.mean(axis=0)
        table[f"sd_true_positive_rate_{class_name}"] = tpr_array.std(axis=0)

    return table


def save_outputs(
    source_data: pd.DataFrame,
    y: np.ndarray,
    result: CrossValidationResult,
) -> None:
    """Save numeric outputs needed to reproduce the reported results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions, classification_metrics = calculate_classification_metrics(
        y,
        result.oof_probabilities,
    )

    identifier_columns = [
        column for column in IDENTIFIER_COLUMNS if column in source_data.columns
    ]
    prediction_table = source_data[identifier_columns].copy().reset_index(drop=True)
    prediction_table["fold"] = result.fold_assignments
    prediction_table["true_label"] = y
    prediction_table["true_class"] = [CLASS_NAMES[value] for value in y]
    for class_index, class_name in enumerate(CLASS_NAMES):
        prediction_table[f"probability_{class_name}"] = result.oof_probabilities[
            :, class_index
        ]
    prediction_table["predicted_label"] = predictions
    prediction_table["predicted_class"] = [
        CLASS_NAMES[value] for value in predictions
    ]
    prediction_table["correct"] = (predictions == y).astype(int)
    prediction_table.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_fold_metrics(result).to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_mean_roc_table(y, result).to_csv(
        OUTPUT_DIR / "mean_roc_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_counts = np.bincount(y, minlength=N_CLASSES)
    metrics: dict[str, object] = {
        "model": "multinomial logistic regression",
        "task": "three-class invasion-depth prediction",
        "classes": list(CLASS_NAMES),
        "class_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
        "n_records": int(len(y)),
        "class_counts": {
            CLASS_NAMES[index]: int(class_counts[index])
            for index in range(N_CLASSES)
        },
        "features": list(FEATURE_ORDER),
        "preprocessing": ["median imputation", "standardization"],
        "model_parameters": LOGISTIC_PARAMETERS,
        "n_splits": N_SPLITS,
        "cv_seed": CV_SEED,
        "model_seed": MODEL_SEED,
        "mean_fold_macro_auc_ovr": result.mean_macro_auc,
        "std_fold_macro_auc_ovr": result.std_macro_auc,
        "mean_fold_macro_average_precision": (
            result.mean_macro_average_precision
        ),
        "std_fold_macro_average_precision": (
            result.std_macro_average_precision
        ),
        "oof_macro_auc_ovr": multiclass_auc(y, result.oof_probabilities),
        "oof_macro_average_precision": multiclass_average_precision(
            y,
            result.oof_probabilities,
        ),
        "fold_macro_aucs": result.fold_macro_aucs,
        "fold_macro_average_precisions": (
            result.fold_macro_average_precisions
        ),
        "per_class_mean_fold_auc": {
            CLASS_NAMES[index]: float(np.mean(result.fold_class_aucs[index]))
            for index in range(N_CLASSES)
        },
        "per_class_std_fold_auc": {
            CLASS_NAMES[index]: float(np.std(result.fold_class_aucs[index]))
            for index in range(N_CLASSES)
        },
        **classification_metrics,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        f"Mean fold macro AUROC={result.mean_macro_auc:.4f} "
        f"+/- {result.std_macro_auc:.4f}."
    )
    print(
        f"Mean fold macro AP={result.mean_macro_average_precision:.4f} "
        f"+/- {result.std_macro_average_precision:.4f}."
    )
    print(
        f"OOF macro AUROC={metrics['oof_macro_auc_ovr']:.4f}; "
        f"OOF macro AP={metrics['oof_macro_average_precision']:.4f}."
    )
    print(
        f"Argmax accuracy={metrics['accuracy']:.4f}; "
        f"balanced accuracy={metrics['balanced_accuracy']:.4f}; "
        f"macro F1={metrics['macro_f1']:.4f}."
    )
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Results were saved to {OUTPUT_DIR}.")


def main() -> None:
    """Run the internal three-class logistic regression evaluation."""

    data_path = prompt_for_data_path()
    print("Logistic regression three-class invasion-depth evaluation")
    print(f"Input workbook: {data_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    source_data, y = load_dataset(data_path)
    features = build_feature_frame(source_data)
    result = run_cross_validation(features, y)
    save_outputs(source_data, y, result)


if __name__ == "__main__":
    main()
