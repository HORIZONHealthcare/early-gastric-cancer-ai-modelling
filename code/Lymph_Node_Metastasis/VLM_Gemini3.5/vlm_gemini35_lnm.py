"""Run Gemini inference for lymph node metastasis classification."""

import base64
import hashlib
import json
import os
import re
import time
import warnings
from io import BytesIO
from pathlib import Path

warnings.filterwarnings('ignore')

import google.generativeai as genai
import numpy as np
import pandas as pd
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score

# Runtime paths and credentials are supplied interactively or through the environment.
EXCEL_PATH      = None
IMAGE_DIR       = None
OUTPUT_DIR      = None
SHEET_NAME      = "final 随访"
IMG_COL_IDX     = 47
LYMPH_COL_IDX   = 69
NAME_COL_IDX    = 2
IMG_COLUMN      = '图'
LYMPH_COLUMN    = 'P淋巴结阳性'
NAME_COLUMN     = '姓名'

GEMINI_API_KEY = ""
INPUT_MODE     = None
GEMINI_MODEL   = "gemini-3.5-flash"

IMAGE_EXTS             = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
REQUEST_DELAY          = 4.5
MAX_IMAGES_PER_PATIENT = 20
PREDICTION_THRESHOLD   = 0.5

MODE_NAMES = {1: 'image_only', 2: 'text_only', 3: 'multimodal'}


def prompt_for_data_path():
    """Request the cohort workbook path from the user."""

    entered_path = input("Input your dataset path: ").strip().strip('"')
    if not entered_path:
        raise ValueError("No dataset path was provided.")
    data_path = Path(entered_path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError("The input workbook was not found.")
    return data_path


def prompt_for_input_mode():
    """Request one of the three input modes used in the experiment."""

    print("Select the Gemini input mode:")
    print("  1 - image only")
    print("  2 - text only")
    print("  3 - multimodal (image + text)")
    entered_mode = input("Input mode [1/2/3]: ").strip()
    if entered_mode not in {'1', '2', '3'}:
        raise ValueError("Input mode must be 1, 2, or 3.")
    return int(entered_mode)


def prompt_for_image_directory():
    """Request the image-root directory for image-based modes."""

    entered_path = input("Input your image directory path: ").strip().strip('"')
    if not entered_path:
        raise ValueError("No image directory path was provided.")
    image_path = Path(entered_path).expanduser()
    if not image_path.is_dir():
        raise NotADirectoryError("The image directory was not found.")
    return image_path


def get_gemini_api_key():
    """Read the credential without embedding or displaying it."""

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Configure it in your environment before running the script."
        )
    return api_key


# Data preparation
THRESHOLDS = {
    'PGI':   {'value': 70,   'dir': 'low'},
    'PGII':  {'value': 21,   'dir': 'high'},
    'PGR':   {'value': 7.5,  'dir': 'low'},
    'CEA':   {'value': 5,    'dir': 'high'},
    'CA199': {'value': 37,   'dir': 'high'},
    'CA125': {'value': 35,   'dir': 'high'},
    'CA153': {'value': 25,   'dir': 'high'},
    'AFP':   {'value': 8.78, 'dir': 'high'},
}
REQUIRED_COLUMNS = {
    '年龄',
    '性别',
    '肿瘤大小提取',
    '分化类类型',
    '层次',
    '脉管浸润',
    '神经束',
    '大体溃疡',
    *THRESHOLDS.keys(),
}


def load_and_validate_dataset(workbook_path):
    """Load the required worksheet and validate every field used by the analysis."""

    with pd.ExcelFile(workbook_path) as workbook:
        if SHEET_NAME not in workbook.sheet_names:
            available = ", ".join(workbook.sheet_names)
            raise ValueError(
                f"Required worksheet '{SHEET_NAME}' was not found. "
                f"Available worksheets: {available}"
            )
        data = pd.read_excel(workbook, sheet_name=SHEET_NAME)

    highest_required_index = max(IMG_COL_IDX, LYMPH_COL_IDX, NAME_COL_IDX)
    if data.shape[1] <= highest_required_index:
        raise ValueError(
            f"Worksheet '{SHEET_NAME}' has {data.shape[1]} columns; "
            f"at least {highest_required_index + 1} are required."
        )

    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Required dataset columns are missing: " + ", ".join(missing_columns)
        )

    indexed_columns = {
        NAME_COL_IDX: NAME_COLUMN,
        IMG_COL_IDX: IMG_COLUMN,
        LYMPH_COL_IDX: LYMPH_COLUMN,
    }
    for column_index, expected_name in indexed_columns.items():
        actual_name = data.columns[column_index]
        if actual_name != expected_name:
            raise ValueError(
                f"Column {column_index + 1} must be '{expected_name}', "
                f"but found '{actual_name}'."
            )

    return data


def clean_numeric(series):
    def parse_value(value):
        value = str(value).strip()
        if value in ['.', '/', 'nan', '', 'None']:
            return np.nan
        value = (value.replace('＜', '').replace('<', '').replace('＞', '').replace('>', '')
                 .replace('≤', '').replace('≥', ''))
        try:
            return float(value)
        except ValueError:
            return np.nan

    return series.apply(parse_value)


def clean_binary(series):
    def parse_value(value):
        value = str(value).strip()
        if value in ['1', '1.0']:
            return 1
        if value in ['0', '0.0']:
            return 0
        return np.nan

    return series.apply(parse_value)


def clean_categorical_value(value):
    value = str(value).strip()
    if value in ['.', '/', 'nan', 'None', '']:
        return 'Unknown'
    return value


def encode_layer_value(value):
    value = str(value).strip()
    if value == '黏膜层':
        return 0
    if value == '黏膜下层':
        return 1
    return np.nan


def binarize_marker_flag(value, threshold, direction):
    numeric_value = clean_numeric(pd.Series([value])).iloc[0]
    if np.isnan(numeric_value):
        return np.nan
    abnormal = (
        numeric_value <= threshold
        if direction == 'low'
        else numeric_value > threshold
    )
    return 1 if abnormal else 0


def compute_impute_stats(data):
    """Compute the cohort-level values used by the original experiment."""

    stats = {}
    age_values = clean_numeric(data['年龄'])
    size_values = clean_numeric(data['肿瘤大小提取'])
    stats['Age'] = float(np.nanmedian(age_values))
    stats['Tumor_Size'] = float(np.nanmedian(size_values))

    for column, key in [
        ('大体溃疡', 'Gross_Ulcer'),
        ('脉管浸润', 'Vascular_Invasion'),
        ('神经束', 'Neural_Invasion'),
    ]:
        flags = clean_binary(data[column])
        stats[key] = int(round(np.nanmedian(flags.astype(float))))

    layer_flags = data['层次'].apply(encode_layer_value)
    stats['Layer'] = int(round(np.nanmedian(layer_flags.astype(float))))

    for column, config in THRESHOLDS.items():
        flags = data[column].apply(
            lambda value: binarize_marker_flag(
                value,
                config['value'],
                config['dir'],
            )
        )
        stats[f'{column}_bin'] = int(round(np.nanmedian(flags.astype(float))))

    return stats


def get_image_paths(name):
    folder = IMAGE_DIR / str(name)
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def update_digest_from_file(digest, file_path):
    """Update a digest from a file without retaining its contents in memory."""

    with file_path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)


def build_run_signature(names):
    """Create a non-identifying signature for safe checkpoint resumption."""

    digest = hashlib.sha256()
    config = {
        'input_mode': INPUT_MODE,
        'model': GEMINI_MODEL,
        'prediction_threshold': PREDICTION_THRESHOLD,
    }
    digest.update(json.dumps(config, sort_keys=True).encode('utf-8'))
    update_digest_from_file(digest, EXCEL_PATH)
    update_digest_from_file(digest, Path(__file__).resolve())

    if INPUT_MODE in (1, 3):
        for sample_index, name in enumerate(names):
            digest.update(f"sample:{sample_index}\n".encode('utf-8'))
            for image_path in get_image_paths(name):
                stat = image_path.stat()
                manifest_entry = (
                    f"{image_path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
                )
                digest.update(manifest_entry.encode('utf-8'))

    return digest.hexdigest()


def load_checkpoint(checkpoint_path, run_signature):
    """Load a checkpoint only when it matches the current run context."""

    if not checkpoint_path.exists():
        return {}

    with checkpoint_path.open('r', encoding='utf-8') as source:
        checkpoint = json.load(source)

    if not isinstance(checkpoint, dict):
        raise ValueError("Existing checkpoint has an invalid top-level structure.")

    if checkpoint.get('run_signature') != run_signature:
        raise ValueError(
            "Existing checkpoint does not match the current dataset, code, or input mode. "
            "Move or delete the existing results directory before starting a new run."
        )

    results = checkpoint.get('results')
    if not isinstance(results, dict):
        raise ValueError("Existing checkpoint has an invalid results structure.")

    return {key: value for key, value in results.items() if value.get('prob') is not None}


def save_checkpoint(checkpoint_path, run_signature, results):
    """Write a checkpoint atomically without storing patient identifiers."""

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    payload = {'run_signature': run_signature, 'results': results}
    with temporary_path.open('w', encoding='utf-8') as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
    temporary_path.replace(checkpoint_path)


def build_clinical_text(row, impute_stats):
    """Build the clinical input while intentionally excluding the target field."""

    lines = ["[Clinical Information]"]

    age = clean_numeric(pd.Series([row.get('年龄', np.nan)])).iloc[0]
    if np.isnan(age):
        age = impute_stats['Age']
    sex_raw = str(row.get('性别', 'Unknown')).strip()
    sex = (
        'Male' if sex_raw in ['男', 'M', '1', '1.0'] else
        'Female' if sex_raw in ['女', 'F', '0', '0.0'] else
        'Unknown'
    )
    lines.append(f"- Age: {age:.0f}")
    lines.append(f"- Sex: {sex}")

    size = clean_numeric(pd.Series([row.get('肿瘤大小提取', np.nan)])).iloc[0]
    if np.isnan(size):
        size = impute_stats['Tumor_Size']
    lines.append(f"- Tumor size: {size:.1f} cm")

    differentiation = clean_categorical_value(row.get('分化类类型', ''))
    lines.append(f"- Differentiation: {differentiation}")

    layer_flag = encode_layer_value(row.get('层次', ''))
    if np.isnan(layer_flag):
        layer_flag = impute_stats['Layer']
    layer_description = (
        'Submucosal invasion (黏膜下层)'
        if layer_flag == 1
        else 'Mucosal layer only (黏膜层)'
    )
    lines.append(f"- Invasion depth: {layer_description}")

    vascular = clean_binary(pd.Series([row.get('脉管浸润', np.nan)])).iloc[0]
    neural = clean_binary(pd.Series([row.get('神经束', np.nan)])).iloc[0]
    ulcer = clean_binary(pd.Series([row.get('大体溃疡', np.nan)])).iloc[0]
    if np.isnan(vascular):
        vascular = impute_stats['Vascular_Invasion']
    if np.isnan(neural):
        neural = impute_stats['Neural_Invasion']
    if np.isnan(ulcer):
        ulcer = impute_stats['Gross_Ulcer']
    lines.append(
        f"- Vascular invasion: {'Positive' if vascular == 1 else 'Negative'}"
    )
    lines.append(
        f"- Neural invasion: {'Positive' if neural == 1 else 'Negative'}"
    )
    lines.append(f"- Gross ulceration: {'Yes' if ulcer == 1 else 'No'}")

    lines.append("- Tumor markers:")
    for column, config in THRESHOLDS.items():
        flag = binarize_marker_flag(
            row.get(column, ''),
            config['value'],
            config['dir'],
        )
        if np.isnan(flag):
            flag = impute_stats[f'{column}_bin']
        lines.append(f"  {column}: {'Abnormal' if flag == 1 else 'Normal'}")

    return "\n".join(lines)


def image_to_base64(path, max_size=512):
    try:
        image = Image.open(path).convert('RGB')
        width, height = image.size
        scale = min(max_size / width, max_size / height, 1.0)
        if scale < 1.0:
            image = image.resize(
                (int(width * scale), int(height * scale)),
                Image.LANCZOS,
            )
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        print("    [Warn] one image could not be loaded")
        return None


# Gemini inference
_PROMPT_BASE = (
    "You are an expert gastroenterologist specializing in early gastric cancer (EGC). "
    "Assess the probability of perigastric lymph node metastasis based on the provided "
    "{input_desc}. "
    "Reply ONLY in this exact format, no extra text:\n"
    "PROBABILITY: <float 0.00-1.00>\n"
    "PREDICTION: <YES if lymph node metastasis is present, NO if absent>\n"
    "REASON: <concise analysis of key risk factors, drawn from {reason_src}>\n"
    "RECOMMENDATION: <must be one of: ESD / Surgery / ESD with close follow-up>"
)

_PROMPT_CFG = {
    1: dict(input_desc="endoscopic images", reason_src="the images"),
    2: dict(input_desc="clinical data", reason_src="the clinical data"),
    3: dict(
        input_desc="endoscopic images and clinical data",
        reason_src="the images and clinical data",
    ),
}


def get_system_prompt():
    return _PROMPT_BASE.format(**_PROMPT_CFG[INPUT_MODE])


def get_generation_config(max_output_tokens=4096):
    return genai.types.GenerationConfig(
        temperature=0.1,
        max_output_tokens=max_output_tokens,
    )


def get_safety_settings():
    return {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


def parse_output(text):
    prob = prediction = reason = recommendation = None

    match = re.search(r'PROBABILITY\s*[:：]\s*([0-9]+\.?[0-9]*)', text, re.I)
    if match:
        try:
            prob = max(0.0, min(1.0, float(match.group(1).rstrip('.'))))
        except ValueError:
            pass

    match = re.search(r'PREDICTION\s*[:：]\s*(YES|NO)', text, re.I)
    if match:
        prediction = match.group(1).upper()
    elif prob is not None:
        prediction = "YES" if prob >= PREDICTION_THRESHOLD else "NO"

    match = re.search(
        r'REASON\s*[:：]\s*(.+?)(?=RECOMMENDATION\s*[:：]|$)',
        text,
        re.I | re.S,
    )
    if match:
        reason = match.group(1).strip()

    match = re.search(r'RECOMMENDATION\s*[:：]\s*(.+)', text, re.I | re.S)
    if match:
        recommendation = match.group(1).strip()

    return prob, prediction, reason or text, recommendation or ""


def query_gemini(model, clinical_text, image_paths):
    parts = []
    loaded = 0

    if INPUT_MODE in (1, 3):
        for path in image_paths[:MAX_IMAGES_PER_PATIENT]:
            encoded_image = image_to_base64(path)
            if encoded_image:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encoded_image,
                    },
                })
                loaded += 1
        image_note = (
            f"The {loaded} endoscopic image(s) above show the gastric lesion."
            if loaded > 0 else "No images could be loaded."
        )
        if INPUT_MODE == 1:
            user_text = (
                f"{image_note}\n"
                "Based on the endoscopic images only, what is the probability of "
                "P lymph node metastasis for this patient?"
            )
        else:
            user_text = (
                "Please assess the lymph node metastasis risk for this early gastric "
                f"cancer patient.\n\n{clinical_text}\n\n"
                f"{image_note}\n"
                "Based on both the images and clinical information, what is the "
                "probability of P lymph node metastasis?"
            )
    else:
        user_text = (
            "Please assess the lymph node metastasis risk for this early gastric cancer "
            f"patient.\n\n{clinical_text}\n\n"
            "Based on the clinical information, what is the probability of P lymph node "
            "metastasis?"
        )

    parts.append({"text": user_text})

    try:
        response = model.generate_content(
            [get_system_prompt()] + parts,
            generation_config=get_generation_config(4096),
            safety_settings=get_safety_settings(),
        )
        try:
            text = response.text.strip()
        except Exception:
            try:
                text = response.candidates[0].content.parts[0].text.strip()
            except Exception:
                # Only multimodal mode uses the original text-only retry.
                if INPUT_MODE == 3:
                    print("    [WARN] image blocked, retrying text-only...")
                    try:
                        text_parts = [{"text": (
                            "Please assess the lymph node metastasis risk for this early "
                            f"gastric cancer patient.\n\n{clinical_text}\n\n"
                            "No endoscopic images available. Based only on the clinical "
                            "information, what is the probability of P lymph node metastasis?"
                        )}]
                        retry_response = model.generate_content(
                            [get_system_prompt()] + text_parts,
                            generation_config=get_generation_config(2048),
                        )
                        text = retry_response.text.strip()
                        print("    [INFO] text-only retry succeeded")
                    except Exception:
                        return None, None, "Blocked", ""
                else:
                    return None, None, "Blocked", ""

        return parse_output(text)
    except Exception as error:
        return None, None, type(error).__name__, ""


def main():
    global EXCEL_PATH, IMAGE_DIR, OUTPUT_DIR, GEMINI_API_KEY, INPUT_MODE

    data_path = prompt_for_data_path()
    INPUT_MODE = prompt_for_input_mode()
    image_path = prompt_for_image_directory() if INPUT_MODE in (1, 3) else None

    EXCEL_PATH = data_path
    IMAGE_DIR = image_path
    OUTPUT_DIR = (
        Path(__file__).resolve().parent /
        f"results_gemini35_lnm_{MODE_NAMES[INPUT_MODE]}"
    )
    GEMINI_API_KEY = get_gemini_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"Initializing Gemini client, model={GEMINI_MODEL}")
    mode_description = {1: 'image-only', 2: 'text-only', 3: 'multimodal'}
    print(f"  INPUT_MODE: {INPUT_MODE}  ({mode_description[INPUT_MODE]})")
    print("  Input workbook: provided")
    if image_path is not None:
        print("  Image directory: provided")
    print(f"  Output directory: {OUTPUT_DIR.name}")

    # Verify API access before processing the cohort.
    print("Testing API connection...")
    try:
        test_response = model.generate_content(
            "Reply with the word OK only.",
            generation_config=get_generation_config(32),
        )
        test_text = test_response.text.strip()
        if not test_text:
            raise ValueError("The API connectivity test returned empty text content.")
        print("API connection: OK")
    except Exception as error:
        print(f"[FATAL] API connection test failed: {type(error).__name__}")
        raise SystemExit(1) from None

    # Load the cohort only after the workbook schema has been validated.
    df_all = load_and_validate_dataset(EXCEL_PATH)
    image_flag = df_all.iloc[:, IMG_COL_IDX]
    lymph_clean = df_all.iloc[:, LYMPH_COL_IDX].astype(str).str.strip()
    mask = (image_flag == 1) & lymph_clean.isin(['0', '1', '0.0', '1.0'])
    df = df_all[mask].copy().reset_index(drop=True)
    labels = df.iloc[:, LYMPH_COL_IDX].astype(str).str.strip().map(
        {'0': 0, '1': 1, '0.0': 0, '1.0': 1}
    ).values
    names = df.iloc[:, NAME_COL_IDX].tolist()

    if df.empty:
        raise ValueError("No eligible lymph-node metastasis samples with images were found.")

    print(f"Valid samples: {len(df)}")
    print(f"  Negative: {(labels == 0).sum()}")
    print(
        f"  Positive: {(labels == 1).sum()}  "
        f"({labels.mean() * 100:.1f}%)"
    )

    impute_stats = compute_impute_stats(df)

    # Resume only when the checkpoint matches this dataset, code, and input mode.
    checkpoint_path = OUTPUT_DIR / 'gemini35_lnm_checkpoint.json'
    run_signature = build_run_signature(names)
    try:
        results = load_checkpoint(checkpoint_path, run_signature)
    except ValueError:
        print(
            "[FATAL] Existing checkpoint is invalid or does not match the current run. "
            "Move or delete the existing results directory before starting again."
        )
        raise SystemExit(1) from None
    if results:
        print(f"[Resume] {len(results)} valid results loaded")

    todo = [
        (sample_index, name, int(label), row)
        for sample_index, (name, label, (_, row))
        in enumerate(zip(names, labels, df.iterrows()))
        if str(sample_index) not in results
    ]
    print(f"Todo: {len(todo)} / {len(names)}  delay={REQUEST_DELAY}s")
    print(f"Estimated time: {len(todo) * REQUEST_DELAY / 60:.1f} min")

    for position, (sample_index, name, label, row) in enumerate(todo):
        result_key = str(sample_index)
        image_paths = get_image_paths(name) if INPUT_MODE in (1, 3) else []
        clinical_text = build_clinical_text(row, impute_stats)

        if INPUT_MODE in (1, 3) and not image_paths:
            print(f"  [{position + 1}/{len(todo)}] no images found, skipping")
            results[result_key] = {
                'label': label,
                'prob': None,
                'prediction_at_0_5': None,
                'reason': '',
                'recommendation': '',
                'n_imgs': 0,
            }
            continue

        prob, _, reason, recommendation = query_gemini(
            model,
            clinical_text,
            image_paths,
        )

        results[result_key] = {
            'label': label,
            'prob': prob,
            'prediction_at_0_5': (
                int(prob >= PREDICTION_THRESHOLD) if prob is not None else None
            ),
            'reason': reason if prob is not None else '',
            'recommendation': recommendation if prob is not None else '',
            'n_imgs': len(image_paths),
        }

        status = f"prob={prob:.3f}" if prob is not None else "FAILED"
        prediction = (
            "N/A" if prob is None else
            "YES" if prob >= PREDICTION_THRESHOLD else "NO"
        )
        print(
            f"  [{position + 1}/{len(todo)}] status={status} "
            f"pred={prediction} imgs={len(image_paths)}"
        )
        if prob is not None:
            print(f"    Reason: {reason}")
            print(f"    Recommendation: {recommendation}")

        if (position + 1) % 10 == 0:
            save_checkpoint(checkpoint_path, run_signature, results)
            print(f"  [Saved] checkpoint {position + 1}")

        time.sleep(REQUEST_DELAY)

    save_checkpoint(checkpoint_path, run_signature, results)

    print(f"\n{'=' * 60}\nEvaluation\n{'=' * 60}")
    valid_keys = [
        str(index) for index in range(len(names))
        if str(index) in results and results[str(index)]['prob'] is not None
    ]
    valid_labels = np.array([results[key]['label'] for key in valid_keys])
    valid_probs = np.array([results[key]['prob'] for key in valid_keys])
    metrics = {
        'threshold': PREDICTION_THRESHOLD,
        'n_total': int(len(names)),
        'n_valid': int(len(valid_keys)),
    }

    if len(valid_keys) < 10:
        print("Not enough valid predictions to evaluate")
    else:
        auc = roc_auc_score(valid_labels, valid_probs)
        average_precision = average_precision_score(valid_labels, valid_probs)
        print(f"AUC    : {auc:.4f}")
        print(f"AP     : {average_precision:.4f}")
        print(f"Decision threshold: {PREDICTION_THRESHOLD:.3f}")
        print(f"Valid  : {len(valid_keys)} / {len(names)}")

        valid_predictions = (valid_probs >= PREDICTION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            valid_labels,
            valid_predictions,
            labels=[0, 1],
        ).ravel()
        accuracy = accuracy_score(valid_labels, valid_predictions)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics.update({
            'auc': float(auc),
            'average_precision': float(average_precision),
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        })
        print(f"\n--- Fixed-threshold ({PREDICTION_THRESHOLD:.1f}) metrics ---")
        print(
            f"Accuracy             : {accuracy:.4f}  "
            f"({int(accuracy * len(valid_labels))}/{len(valid_labels)})"
        )
        print(f"Sensitivity (positive): {sensitivity:.4f}  TP={tp}  FN={fn}")
        print(f"Specificity (negative): {specificity:.4f}  TN={tn}  FP={fp}")
        print(f"PPV                  : {ppv:.4f}")
        print(f"NPV                  : {npv:.4f}")

    predictions_path = OUTPUT_DIR / 'gemini35_vlm_lnm_predictions.csv'
    pd.DataFrame([{
        'sample_index': sample_index,
        'true_label': int(label),
        'gemini_prob_lnm': results.get(str(sample_index), {}).get('prob'),
        'prediction_at_0_5': results.get(str(sample_index), {}).get(
            'prediction_at_0_5'
        ),
        'gemini_reason': results.get(str(sample_index), {}).get('reason', ''),
        'gemini_recommendation': results.get(str(sample_index), {}).get(
            'recommendation', ''
        ),
        'n_imgs': results.get(str(sample_index), {}).get('n_imgs', 0),
    } for sample_index, label in enumerate(labels)]).to_csv(
        predictions_path,
        index=False,
        encoding='utf-8-sig',
    )

    metrics_path = OUTPUT_DIR / 'gemini35_vlm_lnm_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as destination:
        json.dump(metrics, destination, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {OUTPUT_DIR.name}/{predictions_path.name}")
    print(f"[Saved] {OUTPUT_DIR.name}/{checkpoint_path.name}")
    print(f"[Saved] {OUTPUT_DIR.name}/{metrics_path.name}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Stopped] Run interrupted by user.")
        raise SystemExit(130) from None
    except Exception as error:
        print(f"[FATAL] Run failed: {type(error).__name__}")
        raise SystemExit(1) from None
