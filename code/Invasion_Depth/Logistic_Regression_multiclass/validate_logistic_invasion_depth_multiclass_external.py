"""Train the three-class logistic reference model and run external validation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


INTERNAL_SHEET = "final 随访"
EXTERNAL_SHEET = "Sheet1"
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results_logistic_invasion_depth_multiclass_external"
)

SM_COLUMN = "SM"
SM_COLUMN_INDEX = 102
SM_CLASS_MAP = {".": 0, "1": 1, "2": 2, "3": 2}
CLASS_NAMES = ("M", "SM1", "SM2&3")
N_CLASSES = len(CLASS_NAMES)

DIFFERENTIATION_MERGE_MAP = {"未分化型": "未分化", "分化型": "分化"}
FEATURE_ORDER = (
    "Tumor_Size_Bin",
    "Age",
    "Gross_Ulcer",
    "Differentiation",
    "Sex",
)

INTERNAL_COLUMNS = {
    "tumor_size": "肿瘤大小提取",
    "age": "年龄",
    "ulcer": "大体溃疡",
    "differentiation": "分化类类型",
    "sex": "性别",
}

EXTERNAL_COLUMNS = {
    "target": "sm",
    "hospital": "医院",
    "patient_id": "编号",
    "name": "姓名",
    "tumor_size": "大小",
    "age": "年龄",
    "ulcer": "溃疡",
    "differentiation": "分化类型",
    "sex": "性别",
}

MODEL_SEED = 42
LOGISTIC_PARAMETERS: dict[str, object] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": MODEL_SEED,
    "C": 0.1,
    "solver": "lbfgs",
}


def prompt_for_path(prompt: str) -> Path:
    """Read and validate one workbook path."""

    value = input(prompt).strip().strip('"')
    if not value:
        raise ValueError("No dataset path was provided.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    return path


def require_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Load a required worksheet from an Excel workbook."""

    workbook = pd.ExcelFile(path)
    if sheet_name not in workbook.sheet_names:
        raise KeyError(f"Missing required Excel worksheet: {sheet_name!r}")
    return workbook.parse(sheet_name)


def require_columns(
    frame: pd.DataFrame,
    columns: dict[str, str],
    dataset: str,
) -> None:
    """Check the source columns required for modeling."""

    missing = sorted(set(columns.values()).difference(frame.columns))
    if missing:
        raise KeyError(f"Missing columns in the {dataset} workbook: {missing}")


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert source values to numeric values."""

    def parse(value: object) -> float:
        text = str(value).strip()
        if text in {".", "/", "nan", "", "None"}:
            return np.nan
        text = (
            text.replace("＜", "")
            .replace("<", "")
            .replace("＞", "")
            .replace(">", "")
            .replace("≤", "")
            .replace("≥", "")
        )
        try:
            return float(text)
        except ValueError:
            return np.nan

    return series.apply(parse)


def clean_binary(series: pd.Series) -> pd.Series:
    """Convert source values to binary values."""

    def parse(value: object) -> float:
        text = str(value).strip()
        if text in {"1", "1.0"}:
            return 1.0
        if text in {"0", "0.0"}:
            return 0.0
        return np.nan

    return series.apply(parse)


def clean_categorical(series: pd.Series) -> pd.Series:
    """Replace missing categorical values with a stable label."""

    return series.astype(str).replace([".", "/", "nan", "None", ""], "Unknown")


def clean_differentiation(series: pd.Series) -> pd.Series:
    """Normalize differentiation labels shared by both cohorts."""

    return clean_categorical(series).replace(DIFFERENTIATION_MERGE_MAP)


def bin_tumor_size(series: pd.Series) -> pd.Series:
    """Encode tumor size as <=2 cm, >2 to 3 cm, or >3 cm."""

    numeric = clean_numeric(series)

    def encode(value: float) -> float:
        if np.isnan(value):
            return np.nan
        if value <= 2:
            return 0.0
        if value <= 3:
            return 1.0
        return 2.0

    return numeric.apply(encode)


def map_invasion_depth(series: pd.Series) -> pd.Series:
    """Map pathological depth values to M, SM1, and SM2&3."""

    return series.astype(str).str.strip().map(SM_CLASS_MAP)


def load_internal(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the internal development cohort."""

    raw = require_sheet(path, INTERNAL_SHEET)
    require_columns(raw, INTERNAL_COLUMNS, "internal")
    if len(raw.columns) <= SM_COLUMN_INDEX:
        raise ValueError(
            f"The internal worksheet must contain at least "
            f"{SM_COLUMN_INDEX + 1} columns."
        )
    if raw.columns[SM_COLUMN_INDEX] != SM_COLUMN:
        raise ValueError(
            f"Expected column {SM_COLUMN!r} at zero-based position "
            f"{SM_COLUMN_INDEX}, but found {raw.columns[SM_COLUMN_INDEX]!r}."
        )

    raw_target = raw.iloc[:, SM_COLUMN_INDEX].astype(str).str.strip()
    mapped_target = raw_target.map(SM_CLASS_MAP)
    included = mapped_target.notna()
    selected = raw.loc[included].copy().reset_index(drop=True)
    selected["SM_raw"] = raw_target.loc[included].to_numpy()
    target = mapped_target.loc[included].to_numpy(dtype=int)
    return selected, target


def load_external(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the external validation cohort."""

    raw = require_sheet(path, EXTERNAL_SHEET)
    require_columns(raw, EXTERNAL_COLUMNS, "external")
    raw_target = raw[EXTERNAL_COLUMNS["target"]].astype(str).str.strip()
    mapped_target = raw_target.map(SM_CLASS_MAP)
    included = mapped_target.notna()
    selected = raw.loc[included].copy().reset_index(drop=True)
    selected["SM_raw"] = raw_target.loc[included].to_numpy()
    target = mapped_target.loc[included].to_numpy(dtype=int)
    return selected, target


def fit_encoder(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Fit a categorical encoder on the internal cohort."""

    encoder = LabelEncoder()
    values = encoder.fit_transform(series)
    return values, encoder


def transform_encoder(
    series: pd.Series,
    encoder: LabelEncoder,
    column_name: str,
) -> np.ndarray:
    """Apply an internal-cohort encoder to external values."""

    unseen = sorted(set(series.unique()).difference(encoder.classes_))
    if unseen:
        raise ValueError(
            f"External column {column_name!r} contains unseen categories: {unseen}"
        )
    return encoder.transform(series)


def build_internal_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Build the five-feature reference matrix for model training."""

    features = pd.DataFrame(index=frame.index)
    features["Tumor_Size_Bin"] = bin_tumor_size(
        frame[INTERNAL_COLUMNS["tumor_size"]]
    )
    features["Age"] = clean_numeric(frame[INTERNAL_COLUMNS["age"]])
    features["Gross_Ulcer"] = clean_binary(frame[INTERNAL_COLUMNS["ulcer"]])

    differentiation, differentiation_encoder = fit_encoder(
        clean_differentiation(frame[INTERNAL_COLUMNS["differentiation"]])
    )
    sex, sex_encoder = fit_encoder(
        clean_categorical(frame[INTERNAL_COLUMNS["sex"]])
    )
    features["Differentiation"] = differentiation
    features["Sex"] = sex

    encoders = {
        "Differentiation": differentiation_encoder,
        "Sex": sex_encoder,
    }
    return features[list(FEATURE_ORDER)], encoders


def build_external_features(
    frame: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Build the aligned external-validation feature matrix."""

    features = pd.DataFrame(index=frame.index)
    features["Tumor_Size_Bin"] = bin_tumor_size(
        frame[EXTERNAL_COLUMNS["tumor_size"]]
    )
    features["Age"] = clean_numeric(frame[EXTERNAL_COLUMNS["age"]])
    features["Gross_Ulcer"] = clean_binary(frame[EXTERNAL_COLUMNS["ulcer"]])
    features["Differentiation"] = transform_encoder(
        clean_differentiation(frame[EXTERNAL_COLUMNS["differentiation"]]),
        encoders["Differentiation"],
        EXTERNAL_COLUMNS["differentiation"],
    )
    features["Sex"] = transform_encoder(
        clean_categorical(frame[EXTERNAL_COLUMNS["sex"]]),
        encoders["Sex"],
        EXTERNAL_COLUMNS["sex"],
    )
    return features[list(FEATURE_ORDER)]


def multiclass_auc(target: np.ndarray, probability: np.ndarray) -> float:
    """Calculate macro one-vs-rest AUROC."""

    return float(
        roc_auc_score(
            target,
            probability,
            multi_class="ovr",
            average="macro",
        )
    )


def multiclass_average_precision(
    target: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Calculate macro one-vs-rest average precision."""

    binary_target = label_binarize(target, classes=range(N_CLASSES))
    return float(
        average_precision_score(
            binary_target,
            probability,
            average="macro",
        )
    )


def calculate_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calculate the reported external-validation metrics."""

    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(target, prediction, labels=range(N_CLASSES))
    report = classification_report(
        target,
        prediction,
        labels=range(N_CLASSES),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    per_class_auc = {
        CLASS_NAMES[class_index]: float(
            roc_auc_score(
                (target == class_index).astype(int),
                probability[:, class_index],
            )
        )
        for class_index in range(N_CLASSES)
    }

    metrics: dict[str, object] = {
        "decision_rule": "argmax",
        "macro_auc_ovr": multiclass_auc(target, probability),
        "macro_average_precision": multiclass_average_precision(
            target,
            probability,
        ),
        "per_class_auc_ovr": per_class_auc,
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction)
        ),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    return prediction, metrics


def build_roc_table(
    target: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    """Create class-specific external ROC coordinates."""

    tables = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            (target == class_index).astype(int),
            probability[:, class_index],
        )
        tables.append(
            pd.DataFrame(
                {
                    "class": class_name,
                    "point_index": np.arange(len(false_positive_rate)),
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "threshold": thresholds,
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def save_outputs(
    internal_target: np.ndarray,
    external_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    external_target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    validation_metrics: dict[str, object],
    model: LogisticRegression,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    encoders: dict[str, LabelEncoder],
) -> None:
    """Save the model and numeric external-validation results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    identifiers = [
        EXTERNAL_COLUMNS[key]
        for key in ("hospital", "patient_id", "name")
        if EXTERNAL_COLUMNS[key] in external_frame.columns
    ]
    predictions = external_frame[identifiers].copy()
    for feature in FEATURE_ORDER:
        predictions[f"feature_{feature}"] = external_features[
            feature
        ].to_numpy()
    predictions["SM_raw"] = external_frame["SM_raw"].to_numpy()
    predictions["true_label"] = external_target
    predictions["true_class"] = [
        CLASS_NAMES[value] for value in external_target
    ]
    for class_index, class_name in enumerate(CLASS_NAMES):
        predictions[f"probability_{class_name}"] = probability[:, class_index]
    predictions["predicted_label"] = prediction
    predictions["predicted_class"] = [
        CLASS_NAMES[value] for value in prediction
    ]
    predictions["correct"] = (external_target == prediction).astype(int)
    predictions.to_csv(
        OUTPUT_DIR / "external_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_roc_table(external_target, probability).to_csv(
        OUTPUT_DIR / "external_roc_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    internal_counts = np.bincount(internal_target, minlength=N_CLASSES)
    external_counts = np.bincount(external_target, minlength=N_CLASSES)
    metrics: dict[str, object] = {
        "model": "multinomial logistic regression",
        "task": "external validation of three-class invasion-depth prediction",
        "classes": list(CLASS_NAMES),
        "class_mapping": {
            name: index for index, name in enumerate(CLASS_NAMES)
        },
        "n_internal_training_records": int(len(internal_target)),
        "internal_class_counts": {
            CLASS_NAMES[index]: int(internal_counts[index])
            for index in range(N_CLASSES)
        },
        "n_external_validation_records": int(len(external_target)),
        "external_class_counts": {
            CLASS_NAMES[index]: int(external_counts[index])
            for index in range(N_CLASSES)
        },
        "features": list(FEATURE_ORDER),
        "preprocessing": ["median imputation", "standardization"],
        "model_parameters": LOGISTIC_PARAMETERS,
        **validation_metrics,
    }
    with (OUTPUT_DIR / "external_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "encoders": encoders,
            "feature_order": list(FEATURE_ORDER),
            "class_names": list(CLASS_NAMES),
            "decision_rule": "argmax",
            "parameters": LOGISTIC_PARAMETERS,
        },
        OUTPUT_DIR / "logistic_invasion_depth_multiclass_model.joblib",
    )


def main() -> None:
    """Train the reference model and evaluate the external cohort."""

    internal_path = prompt_for_path("Input your internal training dataset path: ")
    external_path = prompt_for_path("Input your external validation dataset path: ")

    internal_frame, internal_target = load_internal(internal_path)
    external_frame, external_target = load_external(external_path)
    internal_features, encoders = build_internal_features(internal_frame)
    external_features = build_external_features(external_frame, encoders)

    imputer = SimpleImputer(strategy="median")
    internal_imputed = imputer.fit_transform(internal_features)
    external_imputed = imputer.transform(external_features)

    scaler = StandardScaler()
    internal_scaled = scaler.fit_transform(internal_imputed)
    external_scaled = scaler.transform(external_imputed)

    model = LogisticRegression(**LOGISTIC_PARAMETERS)
    model.fit(internal_scaled, internal_target)
    external_probability = model.predict_proba(external_scaled)
    external_prediction, metrics = calculate_metrics(
        external_target,
        external_probability,
    )

    save_outputs(
        internal_target,
        external_frame,
        external_features,
        external_target,
        external_probability,
        external_prediction,
        metrics,
        model,
        imputer,
        scaler,
        encoders,
    )

    internal_counts = np.bincount(internal_target, minlength=N_CLASSES)
    external_counts = np.bincount(external_target, minlength=N_CLASSES)
    print("\nLogistic regression three-class invasion-depth external validation")
    print(
        f"Internal cohort: {len(internal_target)} records "
        f"({', '.join(f'{CLASS_NAMES[i]}={int(internal_counts[i])}' for i in range(N_CLASSES))})"
    )
    print(
        f"External cohort: {len(external_target)} records "
        f"({', '.join(f'{CLASS_NAMES[i]}={int(external_counts[i])}' for i in range(N_CLASSES))})"
    )
    print("Model: LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs')")
    print("Decision rule: argmax")
    print(f"Macro AUROC: {metrics['macro_auc_ovr']:.4f}")
    print(f"Macro average precision: {metrics['macro_average_precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Per-class AUROC: {metrics['per_class_auc_ovr']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
