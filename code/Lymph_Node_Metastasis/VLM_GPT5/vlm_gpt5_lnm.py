"""Run GPT-5 inference for lymph node metastasis classification."""

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

import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
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

OPENAI_API_KEY = ""
INPUT_MODE     = None
GPT_MODEL      = "gpt-5"

IMAGE_EXTS             = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
REQUEST_DELAY          = 1.0
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

    print("Select the GPT-5 input mode:")
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


def get_openai_api_key():
    """Read the credential without embedding or displaying it."""

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Configure it in your environment before running the script."
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
    '层次.1',
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


def clean_numeric(val):
    v = str(val).strip()
    if v in ['.', '/', 'nan', '', 'None']:
        return np.nan
    v = (v.replace('＜', '').replace('<', '').replace('＞', '').replace('>', '')
         .replace('≤', '').replace('≥', ''))
    try:
        return float(v)
    except ValueError:
        return np.nan


def clean_binary(val):
    v = str(val).strip()
    if v in ['1', '1.0']:
        return 1
    if v in ['0', '0.0']:
        return 0
    return np.nan


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
        'model': GPT_MODEL,
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


def build_clinical_text(row):
    """Build the clinical input while intentionally excluding the target field."""

    lines = ["[Clinical Information]"]

    age = clean_numeric(row.get('年龄', ''))
    sex_raw = str(row.get('性别', '')).strip()
    sex = ('Male' if sex_raw in ['男', 'M', '1', '1.0'] else
           'Female' if sex_raw in ['女', 'F', '0', '0.0'] else 'Unknown')
    lines.append(f"- Age: {int(age) if not np.isnan(age) else 'Unknown'}")
    lines.append(f"- Sex: {sex}")

    size = clean_numeric(row.get('肿瘤大小提取', ''))
    lines.append(f"- Tumor size: {f'{size:.1f} cm' if not np.isnan(size) else 'Unknown'}")
    lines.append(f"- Differentiation: {str(row.get('分化类类型', '')).strip()}")

    layer = str(row.get('层次.1', '')).strip()
    layer_description = (
        'mucosal' if layer == 'T1a' else
        'submucosal' if layer == 'T1b' else
        'Unknown'
    )
    lines.append(f"- Invasion depth: {layer} ({layer_description})")

    vascular = clean_binary(row.get('脉管浸润', ''))
    neural = clean_binary(row.get('神经束', ''))
    ulcer = clean_binary(row.get('大体溃疡', ''))
    lines.append(
        f"- Vascular invasion: "
        f"{'Positive' if vascular == 1 else 'Negative' if vascular == 0 else 'Unknown'}"
    )
    lines.append(
        f"- Neural invasion: "
        f"{'Positive' if neural == 1 else 'Negative' if neural == 0 else 'Unknown'}"
    )
    lines.append(
        f"- Gross ulceration: "
        f"{'Yes' if ulcer == 1 else 'No' if ulcer == 0 else 'Unknown'}"
    )

    lines.append("- Tumor markers:")
    for col, config in THRESHOLDS.items():
        value = clean_numeric(row.get(col, ''))
        if not np.isnan(value):
            abnormal = (
                value <= config['value'] if config['dir'] == 'low'
                else value > config['value']
            )
            lines.append(f"  {col}: {value:.2f} ({'Abnormal' if abnormal else 'Normal'})")
        else:
            lines.append(f"  {col}: Unknown")

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


# GPT-5 inference
_IMAGE_TEXT_PROMPT_BASE = (
    "You are an expert gastroenterologist specializing in early gastric cancer (EGC). "
    "Assess the probability of perigastric lymph node metastasis based on the provided "
    "{input_desc}. "
    "Reply ONLY in this exact format, no extra text:\n"
    "PROBABILITY: <float 0.00-1.00>\n"
    "PREDICTION: <YES if lymph node metastasis is present, NO if absent>\n"
    "REASON: <concise analysis of key risk factors, drawn from {reason_src}>\n"
    "RECOMMENDATION: <must be one of: ESD / Surgery / ESD with close follow-up>"
)

_IMAGE_TEXT_PROMPT_CFG = {
    1: dict(input_desc="endoscopic images", reason_src="the images"),
    2: dict(input_desc="clinical data", reason_src="the clinical data"),
}

_MULTIMODAL_PROMPT = (
    "You are an expert gastroenterologist specializing in early gastric cancer (EGC). "
    "Assess the probability of perigastric lymph node metastasis based on the provided "
    "endoscopic images and clinical data. "
    "Reply ONLY in this exact format, no extra text:\n"
    "PROBABILITY: <float 0.00-1.00>\n"
    "PREDICTION: <YES if lymph node metastasis is present, NO if absent>\n"
    "REASON: <concise analysis of key risk factors from images and clinical data>\n"
    "RECOMMENDATION: <must be one of: ESD / Surgery / ESD with close follow-up — "
    "choose ESD only if the lesion meets absolute or expanded ESD indications "
    "(differentiated, T1a, <=2cm, no ulcer, or expanded criteria); "
    "choose Surgery (gastrectomy + lymphadenectomy) if submucosal invasion, "
    "undifferentiated, LVI/PNI positive, or high LN metastasis risk; "
    "add a one-sentence rationale>"
)


def get_system_prompt():
    if INPUT_MODE == 3:
        return _MULTIMODAL_PROMPT
    return _IMAGE_TEXT_PROMPT_BASE.format(**_IMAGE_TEXT_PROMPT_CFG[INPUT_MODE])


def call_openai(client, user_content):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=8192,
    )
    message = response.choices[0].message

    if not message.content:
        raise ValueError("The API returned empty content.")
    return message.content.strip()


def build_user_content(clinical_text, image_paths):
    if INPUT_MODE in (1, 3):
        content = []
        n_loaded = 0
        for path in image_paths[:MAX_IMAGES_PER_PATIENT]:
            encoded_image = image_to_base64(path)
            if encoded_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "low",
                    },
                })
                n_loaded += 1

        if n_loaded > 0:
            image_note = f"{n_loaded} endoscopic image(s) attached."
        elif INPUT_MODE == 3:
            image_note = "No images available."
        else:
            image_note = "No images could be loaded."
        if INPUT_MODE == 1:
            content.append({
                "type": "text",
                "text": (
                    f"{image_note}\n\n"
                    "Based on the endoscopic images only, what is the probability of "
                    "perigastric lymph node metastasis for this patient?"
                ),
            })
        else:
            content.append({
                "type": "text",
                "text": (
                    f"Patient data below. {image_note}\n\n"
                    f"{clinical_text}\n\n"
                    "What is the P lymph node metastasis probability?"
                ),
            })
        return content, n_loaded

    content = [{
        "type": "text",
        "text": (
            f"Patient data below.\n\n{clinical_text}\n\n"
            "What is the probability of perigastric lymph node metastasis for this patient?"
        ),
    }]
    return content, 0


def parse_output(text):
    prob = prediction = reason = recommendation = None

    match = re.search(r'PROBABILITY\s*:\s*([0-9]*\.?[0-9]+)', text, re.I)
    if match:
        try:
            prob = max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    match = re.search(r'PREDICTION\s*:\s*(YES|NO)', text, re.I)
    if match:
        prediction = match.group(1).upper()
    elif prob is not None:
        prediction = "YES" if prob >= PREDICTION_THRESHOLD else "NO"

    match = re.search(
        r'REASON\s*:\s*(.+?)(?=RECOMMENDATION\s*:|$)',
        text,
        re.I | re.S,
    )
    if match:
        reason = match.group(1).strip()

    match = re.search(r'RECOMMENDATION\s*:\s*(.+)', text, re.I | re.S)
    if match:
        recommendation = match.group(1).strip()

    return prob, prediction, reason or text, recommendation or ""


def query_gpt5(client, clinical_text, image_paths):
    first_error = None
    try:
        user_content, _ = build_user_content(clinical_text, image_paths)
        raw_output = call_openai(client, user_content)
        return parse_output(raw_output)
    except Exception as error:
        first_error = error
        print(f"    [WARN] attempt 1 failed: {type(error).__name__}")

    # Only multimodal mode falls back to text after the first attempt fails.
    if INPUT_MODE == 3:
        try:
            text_content = [{
                "type": "text",
                "text": (
                    f"Patient data below. No images available.\n\n{clinical_text}\n\n"
                    "What is the P lymph node metastasis probability?"
                ),
            }]
            raw_output = call_openai(client, text_content)
            print("    [INFO] text-only fallback succeeded")
            return parse_output(raw_output)
        except Exception as error:
            print(f"    [ERROR] attempt 2 also failed: {type(error).__name__}")
            return None, None, type(error).__name__, ""

    return None, None, type(first_error).__name__, ""


def main():
    global EXCEL_PATH, IMAGE_DIR, OUTPUT_DIR, OPENAI_API_KEY, INPUT_MODE

    data_path = prompt_for_data_path()
    INPUT_MODE = prompt_for_input_mode()
    image_path = prompt_for_image_directory() if INPUT_MODE in (1, 3) else None

    EXCEL_PATH = data_path
    IMAGE_DIR = image_path
    OUTPUT_DIR = (
        Path(__file__).resolve().parent /
        f"results_gpt5_lnm_{MODE_NAMES[INPUT_MODE]}"
    )
    OPENAI_API_KEY = get_openai_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Initializing OpenAI client, model={GPT_MODEL}")
    mode_description = {1: 'image-only', 2: 'text-only', 3: 'multimodal'}
    print(f"  INPUT_MODE: {INPUT_MODE}  ({mode_description[INPUT_MODE]})")
    print("  Input workbook: provided")
    if image_path is not None:
        print("  Image directory: provided")
    print(f"  Output directory: {OUTPUT_DIR.name}")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Verify API access before processing the cohort.
    print("Testing API connection...")
    try:
        test_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": "Reply with the word OK only."}],
            max_completion_tokens=128,
        )
        if not test_response.choices[0].message.content:
            raise ValueError("The API connectivity test returned empty content.")
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
    labels = lymph_clean[mask].map(
        {'0': 0, '1': 1, '0.0': 0, '1.0': 1}
    ).values
    names = df.iloc[:, NAME_COL_IDX].tolist()

    if df.empty:
        raise ValueError("No eligible lymph-node metastasis samples with images were found.")

    print(f"Valid samples: {len(df)}")
    print(f"  Negative: {(labels == 0).sum()}")
    print(f"  Positive: {(labels == 1).sum()}  ({labels.mean() * 100:.1f}%)")

    # Resume only when the checkpoint matches this dataset, code, and input mode.
    checkpoint_path = OUTPUT_DIR / 'gpt5_lnm_checkpoint.json'
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
        clinical_text = build_clinical_text(row)

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

        prob, _, reason, recommendation = query_gpt5(
            client,
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

    predictions_path = OUTPUT_DIR / 'gpt5_vlm_lnm_predictions.csv'
    pd.DataFrame([{
        'sample_index': sample_index,
        'true_label': int(label),
        'gpt5_prob_lnm': results.get(str(sample_index), {}).get('prob'),
        'prediction_at_0_5': results.get(str(sample_index), {}).get(
            'prediction_at_0_5'
        ),
        'gpt5_reason': results.get(str(sample_index), {}).get('reason', ''),
        'gpt5_recommendation': results.get(str(sample_index), {}).get(
            'recommendation', ''
        ),
        'n_imgs': results.get(str(sample_index), {}).get('n_imgs', 0),
    } for sample_index, label in enumerate(labels)]).to_csv(
        predictions_path,
        index=False,
        encoding='utf-8-sig',
    )

    metrics_path = OUTPUT_DIR / 'gpt5_vlm_lnm_metrics.json'
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
