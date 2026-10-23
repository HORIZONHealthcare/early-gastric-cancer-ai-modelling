"""Train the logistic LNM reference model and run external validation."""

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
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


INTERNAL_SHEET = "final 随访"
EXTERNAL_SHEET = "Sheet1"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_logistic_lnm_external"

PREDICTION_THRESHOLD = 0.5
LOGISTIC_PARAMETERS: dict[str, object] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "C": 0.1,
    "solver": "lbfgs",
}

TARGET_MAP = {"0": 0, "1": 1, "0.0": 0, "1.0": 1}
DIFFERENTIATION_MERGE_MAP = {"未分化型": "未分化", "分化型": "分化"}
FEATURE_ORDER = [
    "Tumor_Size_Bin",
    "Age",
    "Gross_Ulcer",
    "Vascular_Invasion",
    "Neural_Invasion",
    "Layer_T1b",
    "Differentiation",
    "Sex",
]

INTERNAL_COLUMNS = {
    "target": "P淋巴结阳性",
    "tumor_size": "肿瘤大小提取",
    "age": "年龄",
    "ulcer": "大体溃疡",
    "vascular_invasion": "脉管浸润",
    "neural_invasion": "神经束",
    "layer": "层次",
    "differentiation": "分化类类型",
    "sex": "性别",
}

EXTERNAL_COLUMNS = {
    "target": "淋巴结转移",
    "hospital": "医院",
    "patient_id": "编号",
    "name": "姓名",
    "tumor_size": "大小",
    "age": "年龄",
    "ulcer": "溃疡",
    "vascular_invasion": "脉管浸润",
    "neural_invasion": "神经束浸润",
    "layer": "层次",
    "differentiation": "分化类型",
    "sex": "性别",
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


def require_columns(frame: pd.DataFrame, columns: dict[str, str], dataset: str) -> None:
    """Check the source columns required for modeling."""

    missing = sorted(set(columns.values()).difference(frame.columns))
    if missing:
        raise KeyError(f"Missing columns in the {dataset} workbook: {missing}")


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert source values to numeric values."""

    def parse(value: object) -> float:
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

    return series.astype(str).replace([".", "nan", "None", ""], "Unknown")


def clean_differentiation(series: pd.Series) -> pd.Series:
    """Normalize differentiation labels shared by both cohorts."""

    return clean_categorical(series).replace(DIFFERENTIATION_MERGE_MAP)


def encode_invasion_depth(series: pd.Series) -> pd.Series:
    """Encode mucosal and submucosal invasion depth."""

    def encode(value: object) -> float:
        text = str(value).strip()
        if text == "黏膜层":
            return 0.0
        if text == "黏膜下层":
            return 1.0
        return np.nan

    return series.apply(encode)


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


def load_labelled_records(
    path: Path,
    sheet_name: str,
    columns: dict[str, str],
    dataset: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load records with an evaluable lymph-node-metastasis label."""

    raw = pd.read_excel(path, sheet_name=sheet_name)
    require_columns(raw, columns, dataset)
    target_text = raw[columns["target"]].astype(str).str.strip()
    mask = target_text.isin(TARGET_MAP)
    selected = raw.loc[mask].copy().reset_index(drop=True)
    target = target_text.loc[mask].map(TARGET_MAP).to_numpy(dtype=int)
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
    """Build the reference feature matrix for model training."""

    features = pd.DataFrame(index=frame.index)
    features["Tumor_Size_Bin"] = bin_tumor_size(
        frame[INTERNAL_COLUMNS["tumor_size"]]
    )
    features["Age"] = clean_numeric(frame[INTERNAL_COLUMNS["age"]])
    features["Gross_Ulcer"] = clean_binary(frame[INTERNAL_COLUMNS["ulcer"]])
    features["Vascular_Invasion"] = clean_binary(
        frame[INTERNAL_COLUMNS["vascular_invasion"]]
    )
    features["Neural_Invasion"] = clean_binary(
        frame[INTERNAL_COLUMNS["neural_invasion"]]
    )
    features["Layer_T1b"] = encode_invasion_depth(
        frame[INTERNAL_COLUMNS["layer"]]
    )

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
    return features[FEATURE_ORDER], encoders


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
    features["Vascular_Invasion"] = clean_binary(
        frame[EXTERNAL_COLUMNS["vascular_invasion"]]
    )
    features["Neural_Invasion"] = clean_binary(
        frame[EXTERNAL_COLUMNS["neural_invasion"]]
    )
    features["Layer_T1b"] = encode_invasion_depth(
        frame[EXTERNAL_COLUMNS["layer"]]
    )
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
    return features[FEATURE_ORDER]


def calculate_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calculate external-validation metrics at threshold 0.5."""

    prediction = (probability >= PREDICTION_THRESHOLD).astype(int)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    metrics: dict[str, object] = {
        "n": int(len(target)),
        "positive": int(target.sum()),
        "threshold": PREDICTION_THRESHOLD,
        "auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "confusion_matrix": matrix.tolist(),
    }
    return prediction, metrics


def save_outputs(
    external_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, object],
    model: LogisticRegression,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    encoders: dict[str, LabelEncoder],
) -> None:
    """Save the model and external-validation results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    identifiers = [
        EXTERNAL_COLUMNS[key]
        for key in ("hospital", "patient_id", "name")
        if EXTERNAL_COLUMNS[key] in external_frame.columns
    ]
    predictions = external_frame[identifiers].copy()
    for feature in FEATURE_ORDER:
        predictions[f"feature_{feature}"] = external_features[feature].to_numpy()
    predictions["true_label"] = target
    predictions["predicted_probability"] = probability
    predictions["predicted_label"] = prediction
    predictions["correct"] = (target == prediction).astype(int)
    predictions.to_csv(
        OUTPUT_DIR / "external_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (OUTPUT_DIR / "external_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "encoders": encoders,
            "feature_order": FEATURE_ORDER,
            "threshold": PREDICTION_THRESHOLD,
            "parameters": LOGISTIC_PARAMETERS,
        },
        OUTPUT_DIR / "logistic_lnm_reference_model.joblib",
    )


def main() -> None:
    """Run model training and external validation."""

    internal_path = prompt_for_path("Input your internal training dataset path: ")
    external_path = prompt_for_path("Input your external validation dataset path: ")

    internal_frame, internal_target = load_labelled_records(
        internal_path,
        INTERNAL_SHEET,
        INTERNAL_COLUMNS,
        "internal",
    )
    external_frame, external_target = load_labelled_records(
        external_path,
        EXTERNAL_SHEET,
        EXTERNAL_COLUMNS,
        "external",
    )
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
    external_probability = model.predict_proba(external_scaled)[:, 1]
    external_prediction, metrics = calculate_metrics(
        external_target,
        external_probability,
    )

    save_outputs(
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

    print("\nLogistic regression lymph-node-metastasis external validation")
    print(
        f"Internal cohort: {len(internal_target)} records "
        f"({int(internal_target.sum())} LNM-positive)"
    )
    print(
        f"External cohort: {metrics['n']} records "
        f"({metrics['positive']} LNM-positive)"
    )
    print("Model: LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs')")
    print(f"AUROC: {metrics['auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
