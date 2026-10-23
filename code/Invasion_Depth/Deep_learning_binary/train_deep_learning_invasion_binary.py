"""Train the deep learning model for binary invasion-depth classification."""

import hashlib
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import open_clip

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Runtime paths are provided interactively.
EXCEL_PATH = None
IMAGE_DIR = None
OUTPUT_ROOT = None
FOLD_SPLIT_PATH = None

SHEET_NAME    = "final 随访"
IMG_COL_IDX   = 47
LAYER_COL_IDX = 56
NAME_COL_IDX  = 2
IMG_COLUMN      = '图'
LAYER_COLUMN    = '层次'
NAME_COLUMN     = '姓名'

MODE_NAMES = {
    'image-only': 'image_only',
    'tabular-only': 'tabular_only',
    'multimodal': 'multimodal',
}
MODEL_MODE = None

N_EPOCHS       = 40
N_FOLDS        = 5
IMG_BATCH_SIZE = 8
TAB_BATCH_SIZE = 64
LR             = 3e-4

IMG_SIZE          = 224
RANDOM_SEED       = 42
LABEL_SMOOTH      = 0.1
FOCAL_GAMMA       = 2.0
FEAT_DIM          = 512    # BiomedCLIP ViT-B/16 CLS output dimension

BACKBONE_LR_RATIO = 0.005  # backbone LR = LR * ratio (low to prevent forgetting)
PREDICTION_THRESHOLD = 0.5

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def prompt_for_data_path():
    entered_path = input("Input your dataset path: ").strip().strip('"')
    if not entered_path:
        raise ValueError("No dataset path was provided.")
    data_path = Path(entered_path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError("The input workbook was not found.")
    return data_path


def prompt_for_image_directory():
    entered_path = input("Input your image directory path: ").strip().strip('"')
    if not entered_path:
        raise ValueError("No image directory path was provided.")
    image_path = Path(entered_path).expanduser()
    if not image_path.is_dir():
        raise NotADirectoryError("The image directory was not found.")
    return image_path


def prompt_for_model_mode():
    print("Select the deep learning input mode:")
    print("  1 - image only")
    print("  2 - tabular only")
    print("  3 - multimodal (image + tabular)")
    selected = input("Input mode [1/2/3]: ").strip()
    choices = {'1': 'image-only', '2': 'tabular-only', '3': 'multimodal'}
    if selected not in choices:
        raise ValueError("Input mode must be 1, 2, or 3.")
    return choices[selected]

# ============================================================
# Image transforms
# ============================================================
# Training: augmented
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Validation: standard (no augmentation)
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================
# 1. Data cleaning helpers
# ============================================================
THRESHOLDS = {
    'PGI':   {'value': 70,   'dir': 'low',  'label': 'PGI_bin'},
    'PGII':  {'value': 21,   'dir': 'high', 'label': 'PGII_bin'},
    'PGR':   {'value': 7.5,  'dir': 'low',  'label': 'PGR_bin'},
    'CEA':   {'value': 5,    'dir': 'high', 'label': 'CEA_bin'},
    'CA199': {'value': 37,   'dir': 'high', 'label': 'CA199_bin'},
    'CA125': {'value': 35,   'dir': 'high', 'label': 'CA125_bin'},
    'CA153': {'value': 25,   'dir': 'high', 'label': 'CA153_bin'},
    'AFP':   {'value': 8.78, 'dir': 'high', 'label': 'AFP_bin'},
}
CATEGORICAL_COLS = ['分化类类型', '性别']
FEATURE_NAME_MAP = {'分化类类型': 'Differentiation', '性别': 'Sex'}
REQUIRED_COLUMNS = {
    '分化类类型',
    '性别',
    '肿瘤大小提取',
    '年龄',
    '大体溃疡',
    '层次',
    *THRESHOLDS.keys(),
}


def load_and_validate_dataset(workbook_path):
    with pd.ExcelFile(workbook_path) as workbook:
        if SHEET_NAME not in workbook.sheet_names:
            raise ValueError(f"Required worksheet '{SHEET_NAME}' was not found.")
        data = pd.read_excel(workbook, sheet_name=SHEET_NAME)

    highest_required_index = max(IMG_COL_IDX, LAYER_COL_IDX, NAME_COL_IDX)
    if data.shape[1] <= highest_required_index:
        raise ValueError(
            f"Worksheet '{SHEET_NAME}' must contain at least "
            f"{highest_required_index + 1} columns."
        )

    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Required dataset columns are missing: " + ", ".join(missing_columns)
        )

    indexed_columns = {
        NAME_COL_IDX: NAME_COLUMN,
        IMG_COL_IDX: IMG_COLUMN,
        LAYER_COL_IDX: LAYER_COLUMN,
    }
    for column_index, expected_name in indexed_columns.items():
        if data.columns[column_index] != expected_name:
            raise ValueError(
                f"Column {column_index + 1} must be '{expected_name}'."
            )

    return data


def update_digest_from_file(digest, file_path):
    with Path(file_path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)


def build_run_signature(labels, image_paths):
    digest = hashlib.sha256()
    configuration = {
        'task': 'binary_invasion_depth',
        'model': 'BiomedCLIP_ViT_B_16',
        'n_folds': N_FOLDS,
        'n_epochs': N_EPOCHS,
        'random_seed': RANDOM_SEED,
        'prediction_threshold': PREDICTION_THRESHOLD,
    }
    digest.update(json.dumps(configuration, sort_keys=True).encode('utf-8'))
    update_digest_from_file(digest, EXCEL_PATH)
    update_digest_from_file(digest, Path(__file__).resolve())
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())

    for sample_index, paths in enumerate(image_paths):
        digest.update(f"sample:{sample_index}\n".encode('utf-8'))
        for image_path in paths:
            stat = Path(image_path).stat()
            entry = f"{Path(image_path).name}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
            digest.update(entry.encode('utf-8'))

    return digest.hexdigest()


def save_json_atomic(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def save_torch_atomic(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def clean_numeric(series):
    def _parse(v):
        v = str(v).strip()
        if v in ['.', '/', 'nan', '', 'None']:
            return np.nan
        v = (v.replace('＜', '').replace('<', '').replace('＞', '').replace('>', '')
              .replace('≤', '').replace('≥', ''))
        try:
            return float(v)
        except Exception:
            return np.nan
    return series.apply(_parse)


def clean_binary(series):
    def _parse(v):
        v = str(v).strip()
        if v in ['1', '1.0']: return 1
        if v in ['0', '0.0']: return 0
        return np.nan
    return series.apply(_parse)


def clean_categorical(series):
    return series.astype(str).replace(['.', '/', 'nan', 'None', ''], 'Unknown')


def binarize_marker(series, threshold, direction):
    numeric = clean_numeric(series)
    def _bin(v):
        if np.isnan(v): return np.nan
        return 1 if (v <= threshold if direction == 'low' else v > threshold) else 0
    return numeric.apply(_bin)


def build_raw_tabular(df):
    """
    Extract raw tabular features for binary invasion-depth classification.
    Returns a DataFrame with NaN preserved.

    Post-operative vascular and neural invasion are excluded, and invasion
    depth itself is not used as an input feature.
    """
    X = pd.DataFrame(index=df.index)
    for col, cfg in THRESHOLDS.items():
        X[cfg['label']] = binarize_marker(df[col], cfg['value'], cfg['dir'])
    X['Tumor_Size']  = clean_numeric(df['肿瘤大小提取'])
    X['Age']         = clean_numeric(df['年龄'])
    X['Gross_Ulcer'] = clean_binary(df['大体溃疡'])
    for col in CATEGORICAL_COLS:
        X[FEATURE_NAME_MAP[col]] = clean_categorical(df[col])
    return X


def fit_fold_preprocessors(X_raw_train):
    """
    Fit imputer, scaler, and label encoders on training fold raw features.
    Returns (X_train_scaled, preprocessors_dict).
    """
    X = X_raw_train.copy()

    # Label-encode categorical columns first (they stay as strings in raw)
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        name = FEATURE_NAME_MAP[col]
        le   = LabelEncoder()
        X[name] = le.fit_transform(X[name].astype(str))
        label_encoders[col] = le

    X_vals = X.values.astype(float)

    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_imp   = imputer.fit_transform(X_vals)
    X_sc    = scaler.fit_transform(X_imp)

    preprocessors = {
        'imputer':        imputer,
        'scaler':         scaler,
        'label_encoders': label_encoders,
    }
    return X_sc, preprocessors


def transform_fold(X_raw_val, preprocessors):
    """
    Apply pre-fitted preprocessors to val fold.
    """
    X = X_raw_val.copy()
    le_dict = preprocessors['label_encoders']

    for col in CATEGORICAL_COLS:
        name  = FEATURE_NAME_MAP[col]
        le    = le_dict[col]
        known = set(le.classes_)
        X[name] = X[name].astype(str).apply(
            lambda v: v if v in known else le.classes_[0])
        X[name] = le.transform(X[name])

    X_vals = X.values.astype(float)
    X_imp  = preprocessors['imputer'].transform(X_vals)
    X_sc   = preprocessors['scaler'].transform(X_imp)
    return X_sc


# ============================================================
# 2. Image utilities
# ============================================================
def get_image_paths(name, image_dir):
    folder = Path(image_dir) / str(name)
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


# ============================================================
# 3. Datasets
# ============================================================
class ImageDataset(Dataset):
    """Direct image loading with separate train / val transforms."""
    def __init__(self, names, labels, img_paths_list, transform):
        self.names     = names
        self.labels    = labels
        self.img_paths = img_paths_list
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = []
        for p in self.img_paths[idx]:
            try:
                imgs.append(self.transform(Image.open(p).convert('RGB')))
            except Exception:
                continue
        if not imgs:
            imgs = [torch.zeros(3, IMG_SIZE, IMG_SIZE)]
        return torch.stack(imgs), torch.tensor(self.labels[idx], dtype=torch.float32)


class TabularDataset(Dataset):
    def __init__(self, tab_arr, labels):
        self.tab    = torch.FloatTensor(tab_arr)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tab[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


class MultimodalDataset(Dataset):
    """Combined image + tabular dataset."""
    def __init__(self, tab_arr, names, labels, img_paths_list, transform):
        self.tab       = torch.FloatTensor(tab_arr)
        self.names     = names
        self.labels    = labels
        self.img_paths = img_paths_list
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = []
        for p in self.img_paths[idx]:
            try:
                imgs.append(self.transform(Image.open(p).convert('RGB')))
            except Exception:
                continue
        if not imgs:
            imgs = [torch.zeros(3, IMG_SIZE, IMG_SIZE)]
        return (self.tab[idx], torch.stack(imgs),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def image_collate_fn(batch):
    """Pad variable-length image sequences."""
    imgs_list, labels = zip(*batch)
    max_n  = max(imgs.shape[0] for imgs in imgs_list)
    B      = len(imgs_list)
    padded = torch.zeros(B, max_n, 3, IMG_SIZE, IMG_SIZE)
    masks  = torch.zeros(B, max_n, dtype=torch.bool)
    for i, imgs in enumerate(imgs_list):
        n = imgs.shape[0]
        padded[i, :n] = imgs
        masks[i, :n]  = True
    return padded, masks, torch.stack(labels)


def multimodal_collate_fn(batch):
    tabs, imgs_list, labels = zip(*batch)
    max_n  = max(imgs.shape[0] for imgs in imgs_list)
    B      = len(imgs_list)
    padded = torch.zeros(B, max_n, 3, IMG_SIZE, IMG_SIZE)
    masks  = torch.zeros(B, max_n, dtype=torch.bool)
    for i, imgs in enumerate(imgs_list):
        n = imgs.shape[0]
        padded[i, :n] = imgs
        masks[i, :n]  = True
    return torch.stack(tabs), padded, masks, torch.stack(labels)


# ============================================================
# 4. BiomedCLIP loader
# ============================================================
def load_biomedclip_encoder(device):
    """
    Load BiomedCLIP ViT-B/16, fully unfreeze all parameters.
    Called fresh for each fold and each mode - no weight sharing.
    """
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    encoder = clip_model.visual.to(device)

    # Unfreeze everything
    for p in encoder.parameters():
        p.requires_grad = True

    trunk = encoder.trunk
    # Make sure patch embed and pos embed are trainable
    if hasattr(trunk, 'patch_embed'):
        for p in trunk.patch_embed.parameters():
            p.requires_grad = True
    if hasattr(trunk, 'pos_embed') and trunk.pos_embed is not None:
        trunk.pos_embed.requires_grad = True
    if hasattr(trunk, 'cls_token') and trunk.cls_token is not None:
        trunk.cls_token.requires_grad = True

    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"    BiomedCLIP loaded - all {n_total/1e6:.1f}M params trainable")
    return encoder


# ============================================================
# 5. Model definitions
# ============================================================
class AttentionPooling(nn.Module):
    """Attention pooling over a set of image features."""
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, feats, mask):
        """
        feats : (B, N, feat_dim)
        mask  : (B, N) bool
        Returns pooled (B, feat_dim), weights (B, N)
        """
        scores = self.attn(feats).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        w      = torch.softmax(scores, dim=1)
        pooled = (feats * w.unsqueeze(-1)).sum(dim=1)
        return pooled, w


class ImageBranch(nn.Module):
    """BiomedCLIP encoder + attention pooling + projection."""
    def __init__(self, encoder, feat_dim=FEAT_DIM, out_dim=256, dropout=0.4):
        super().__init__()
        self.encoder   = encoder
        self.attn_pool = AttentionPooling(feat_dim)
        self.proj      = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, out_dim), nn.GELU())

    def forward(self, imgs, mask):
        """
        imgs : (B, N, 3, H, W)
        mask : (B, N) bool
        Returns projected feature (B, out_dim), attention weights (B, N)
        """
        B, N, C, H, W = imgs.shape
        # All images encoded in one pass; gradient flows through encoder
        feats          = self.encoder(imgs.view(B * N, C, H, W)).view(B, N, -1)
        pooled, w      = self.attn_pool(feats, mask)
        return self.proj(pooled), w


class TabularBranch(nn.Module):
    def __init__(self, input_dim, out_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, img_dim, tab_dim):
        super().__init__()
        self.tab_proj = nn.Linear(tab_dim, img_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + tab_dim, img_dim), nn.Sigmoid())

    def forward(self, img_feat, tab_feat):
        tab_proj = self.tab_proj(tab_feat)
        gate     = self.gate_net(torch.cat([img_feat, tab_feat], dim=1))
        return gate * img_feat + (1 - gate) * tab_proj


class ImageOnlyModel(nn.Module):
    """BiomedCLIP (full unfreeze) + attention pooling + classifier."""
    def __init__(self, encoder, img_out_dim=256, dropout=0.4):
        super().__init__()
        self.img_branch = ImageBranch(encoder, FEAT_DIM, img_out_dim, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(img_out_dim), nn.Dropout(dropout * 0.5),
            nn.Linear(img_out_dim, 64), nn.GELU(),
            nn.Dropout(dropout * 0.25), nn.Linear(64, 1))

    def forward(self, imgs, mask):
        feat, attn_w = self.img_branch(imgs, mask)
        return self.classifier(feat).squeeze(1), attn_w

    def get_backbone_params(self):
        return [p for p in self.img_branch.encoder.parameters() if p.requires_grad]

    def get_head_params(self):
        mods = [self.img_branch.attn_pool, self.img_branch.proj, self.classifier]
        return [p for m in mods for p in m.parameters() if p.requires_grad]

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TabularOnlyModel(nn.Module):
    """TabularBranch + classifier."""
    def __init__(self, input_dim, tab_out_dim=64, dropout=0.4):
        super().__init__()
        self.tab_branch = TabularBranch(input_dim, tab_out_dim, dropout * 0.75)
        self.classifier = nn.Sequential(
            nn.LayerNorm(tab_out_dim), nn.Dropout(dropout * 0.5),
            nn.Linear(tab_out_dim, 32), nn.GELU(),
            nn.Dropout(dropout * 0.25), nn.Linear(32, 1))

    def forward(self, tab):
        return self.classifier(self.tab_branch(tab)).squeeze(1)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultimodalModel(nn.Module):
    """Image branch + tabular branch + gated fusion + classifier."""
    def __init__(self, encoder, tab_input_dim, img_out_dim=256, tab_out_dim=64, dropout=0.4):
        super().__init__()
        self.img_branch  = ImageBranch(encoder, FEAT_DIM, img_out_dim, dropout)
        self.tab_branch  = TabularBranch(tab_input_dim, tab_out_dim, dropout * 0.75)
        self.gate_fusion = GatedFusion(img_out_dim, tab_out_dim)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(img_out_dim), nn.Dropout(dropout * 0.5),
            nn.Linear(img_out_dim, 64), nn.GELU(),
            nn.Dropout(dropout * 0.25), nn.Linear(64, 1))

    def forward(self, tab, imgs, mask):
        img_feat, attn_w = self.img_branch(imgs, mask)
        tab_feat         = self.tab_branch(tab)
        fused            = self.gate_fusion(img_feat, tab_feat)
        return self.classifier(fused).squeeze(1), attn_w

    def get_backbone_params(self):
        return [p for p in self.img_branch.encoder.parameters() if p.requires_grad]

    def get_head_params(self):
        mods = [self.img_branch.attn_pool, self.img_branch.proj,
                self.tab_branch, self.gate_fusion, self.classifier]
        return [p for m in mods for p in m.parameters() if p.requires_grad]

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 6. Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, label_smooth=0.1):
        super().__init__()
        self.gamma        = gamma
        self.pos_weight   = pos_weight
        self.label_smooth = label_smooth

    def forward(self, logits, targets):
        targets_s = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_s, pos_weight=self.pos_weight, reduction='none')
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


# ============================================================
# 7. Train / eval loops (one set per mode)
# ============================================================
def make_pos_weight(y_tr):
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)


# -- image-only --
def train_epoch_image(model, loader, optimizer, criterion, device, desc=''):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=desc, leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    for imgs, masks, labels in pbar:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs, masks)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch_image(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    for imgs, masks, labels in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits, _   = model(imgs, masks)
        probs_all.extend(torch.sigmoid(logits).cpu().numpy())
        labels_all.extend(labels.numpy())
    return np.array(probs_all), np.array(labels_all)


# -- tabular-only --
def train_epoch_tabular(model, loader, optimizer, criterion, device, desc=''):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=desc, leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    for tab, labels in pbar:
        tab, labels = tab.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(tab)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch_tabular(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    for tab, labels in loader:
        tab    = tab.to(device)
        logits = model(tab)
        probs_all.extend(torch.sigmoid(logits).cpu().numpy())
        labels_all.extend(labels.numpy())
    return np.array(probs_all), np.array(labels_all)


# -- multimodal --
def train_epoch_multimodal(model, loader, optimizer, criterion, device, desc=''):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=desc, leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    for tab, imgs, masks, labels in pbar:
        tab, imgs, masks, labels = (tab.to(device), imgs.to(device),
                                    masks.to(device), labels.to(device))
        optimizer.zero_grad()
        logits, _ = model(tab, imgs, masks)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch_multimodal(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    for tab, imgs, masks, labels in loader:
        tab, imgs, masks = tab.to(device), imgs.to(device), masks.to(device)
        logits, _        = model(tab, imgs, masks)
        probs_all.extend(torch.sigmoid(logits).cpu().numpy())
        labels_all.extend(labels.numpy())
    return np.array(probs_all), np.array(labels_all)


# ============================================================
# 8. Attention weight export
# ============================================================
def save_attention_weights_image(model, val_dl, val_indices, device,
                                 save_dir, fold, val_probs):
    """Save deidentified attention weights for image-based models."""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    all_weights  = {}
    sample_cursor = 0

    with torch.no_grad():
        for batch in val_dl:
            if len(batch) == 3:
                imgs_b, masks_b, labels_b = batch
                tab_b = None
            else:
                tab_b, imgs_b, masks_b, labels_b = batch

            imgs_b  = imgs_b.to(device)
            masks_b = masks_b.to(device)

            if tab_b is not None:
                tab_b = tab_b.to(device)
                _, attn_w = model(tab_b, imgs_b, masks_b)
            else:
                _, attn_w = model(imgs_b, masks_b)

            for i in range(imgs_b.size(0)):
                sample_index = int(val_indices[sample_cursor])
                valid = masks_b[i].cpu().numpy().astype(bool)
                w     = attn_w[i].cpu().numpy()[valid]
                all_weights[str(sample_index)] = {
                    'sample_index': sample_index,
                    'attn_weights_per_img': w.tolist(),
                    'label':                int(labels_b[i].item()),
                    'pred_prob':            float(val_probs[sample_cursor]),
                }
                sample_cursor += 1

    save_path = save_dir / f'attention_weights_fold{fold}.pt'
    save_torch_atomic(save_path, all_weights)
    print(f'    [Saved] attention weights for fold {fold}')


# ============================================================
# 9. Evaluation helpers
# ============================================================
def compute_metrics(y_true, oof_probs, fold_aucs):
    oof_auc = roc_auc_score(y_true, oof_probs)
    pr_auc  = average_precision_score(y_true, oof_probs)
    fpr, tpr, _ = roc_curve(y_true, oof_probs)
    oof_preds = (oof_probs >= PREDICTION_THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, oof_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity    = tp / max(tp + fn, 1)
    specificity    = tn / max(tn + fp, 1)

    return {
        'oof_auc':        oof_auc,
        'pr_auc':         pr_auc,
        'mean_fold_auc':  np.mean(fold_aucs),
        'std_fold_auc':   np.std(fold_aucs),
        'threshold':      PREDICTION_THRESHOLD,
        'accuracy':       accuracy_score(y_true, oof_preds),
        'balanced_accuracy': balanced_accuracy_score(y_true, oof_preds),
        'f1':              f1_score(y_true, oof_preds, zero_division=0),
        'sensitivity':    sensitivity,
        'specificity':    specificity,
        'fpr': fpr, 'tpr': tpr,
        'oof_preds': oof_preds,
        'cm': cm,
    }


def build_mean_roc_table(y, oof_probs, splits):
    from scipy.interpolate import interp1d as _interp1d
    common_fpr  = np.linspace(0, 1, 500)
    tprs_interp = []

    for sp in splits:
        val_idx   = sp['val']
        y_v       = np.array([y[i] for i in val_idx])
        p_v       = oof_probs[val_idx]
        fpr_f, tpr_f, _ = roc_curve(y_v, p_v)
        fn = _interp1d(fpr_f, tpr_f, kind='linear',
                       bounds_error=False, fill_value=(0.0, 1.0))
        tprs_interp.append(fn(common_fpr))

    tprs_arr = np.array(tprs_interp)
    return pd.DataFrame({
        'false_positive_rate': common_fpr,
        'mean_true_positive_rate': tprs_arr.mean(axis=0),
        'sd_true_positive_rate': tprs_arr.std(axis=0),
    })


# ============================================================
# 10. Fold training functions
# ============================================================
def train_image_only(y, names, img_paths, fold_splits, output_dir):
    print(f"\n{'='*60}")
    print("MODE: image-only")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    oof_probs    = np.zeros(len(y))
    fold_aucs    = []
    fold_history = []

    for fold, sp in enumerate(fold_splits, 1):
        tr_idx = sp['train']
        val_idx = sp['val']
        print(f"\n-- Fold {fold} --")

        y_tr   = y[tr_idx]
        y_val  = y[val_idx]
        n_tr   = [names[i] for i in tr_idx]
        n_val  = [names[i] for i in val_idx]
        ip_tr  = [img_paths[i] for i in tr_idx]
        ip_val = [img_paths[i] for i in val_idx]

        tr_ds  = ImageDataset(n_tr,  y_tr,  ip_tr,  TRAIN_TRANSFORM)
        val_ds = ImageDataset(n_val, y_val, ip_val, VAL_TRANSFORM)

        class_counts = np.bincount(y_tr.astype(int))
        weights      = 1.0 / class_counts[y_tr.astype(int)]
        sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

        _bs   = min(IMG_BATCH_SIZE, max(1, len(n_tr) // 2))
        _drop = len(n_tr) >= IMG_BATCH_SIZE * 2
        tr_dl  = DataLoader(tr_ds,  batch_size=_bs, sampler=sampler,
                            num_workers=4, collate_fn=image_collate_fn,
                            pin_memory=True, drop_last=_drop)
        val_dl = DataLoader(val_ds, batch_size=max(1, IMG_BATCH_SIZE // 2),
                            shuffle=False, num_workers=4,
                            collate_fn=image_collate_fn, pin_memory=True)

        criterion = FocalLoss(FOCAL_GAMMA, make_pos_weight(y_tr), LABEL_SMOOTH)
        encoder   = load_biomedclip_encoder(DEVICE)
        model     = ImageOnlyModel(encoder).to(DEVICE)

        bb_lr     = LR * BACKBONE_LR_RATIO
        optimizer = torch.optim.AdamW([
            {'params': model.get_backbone_params(), 'lr': bb_lr,  'weight_decay': 0.1},
            {'params': model.get_head_params(),     'lr': LR,     'weight_decay': 1e-2},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS, eta_min=LR * 0.005)

        bb_p   = sum(p.numel() for p in model.get_backbone_params())
        head_p = sum(p.numel() for p in model.get_head_params())
        print(f"  Trainable: backbone {bb_p/1e6:.2f}M @ lr={bb_lr:.2e}, "
              f"head {head_p/1e3:.1f}K @ lr={LR:.2e}")

        # Initialise best_auc to -inf so any real AUC beats it
        best_auc   = -np.inf
        best_state = None
        best_probs = None
        history    = {'loss': [], 'val_auc': []}

        ep_pbar = tqdm(range(1, N_EPOCHS + 1), desc=f"  Fold{fold}",
                       bar_format='{l_bar}{bar:30}{r_bar}')
        for ep in ep_pbar:
            loss = train_epoch_image(model, tr_dl, optimizer, criterion, DEVICE,
                                     desc=f"Ep{ep:>3}")
            scheduler.step()
            probs_v, lbls_v = eval_epoch_image(model, val_dl, DEVICE)
            try:
                auc = roc_auc_score(lbls_v, probs_v)
            except Exception:
                auc = 0.5

            history['loss'].append(loss)
            history['val_auc'].append(auc)

            if auc > best_auc:
                best_auc   = auc
                best_probs = probs_v.copy()
                # Save state to CPU to avoid holding extra VRAM
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_torch_atomic(
                    Path(output_dir) / f'fold{fold}_image_only_best_ckpt.pth', {
                    'epoch':         ep,
                    'fold':          fold,
                    'model_state':   best_state,
                    'val_auc':       auc,
                    'feat_dim':      FEAT_DIM,
                    'img_out_dim':   256,
                    'val_sample_indices': [int(index) for index in val_idx],
                    'val_labels':    y_val.tolist(),
                })

            ep_pbar.set_postfix({'loss': f'{loss:.4f}', 'auc': f'{auc:.4f}',
                                 'best': f'{best_auc:.4f}'})

        # Save deidentified attention weights using the best model.
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        save_attention_weights_image(
            model, val_dl, val_idx, DEVICE,
            save_dir=Path(output_dir) / 'attention_weights',
            fold=fold, val_probs=best_probs)

        oof_probs[val_idx] = best_probs
        fold_aucs.append(best_auc)
        fold_history.append(history)
        print(f"  Fold {fold} best AUC = {best_auc:.4f}")

        # Free VRAM
        del model, encoder, optimizer
        torch.cuda.empty_cache()

    return oof_probs, fold_aucs, fold_history


def train_tabular_only(y, names, X_raw, fold_splits, output_dir):
    print(f"\n{'='*60}")
    print("MODE: tabular-only")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    oof_probs    = np.zeros(len(y))
    fold_aucs    = []
    fold_history = []

    for fold, sp in enumerate(fold_splits, 1):
        tr_idx  = sp['train']
        val_idx = sp['val']
        print(f"\n-- Fold {fold} --")

        y_tr  = y[tr_idx]
        y_val = y[val_idx]

        # Fit preprocessors on training fold only
        X_tr_raw  = X_raw.iloc[tr_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        X_tr_sc, preproc = fit_fold_preprocessors(X_tr_raw)
        X_val_sc         = transform_fold(X_val_raw, preproc)
        INPUT_DIM        = X_tr_sc.shape[1]

        tr_ds  = TabularDataset(X_tr_sc,  y_tr)
        val_ds = TabularDataset(X_val_sc, y_val)

        class_counts = np.bincount(y_tr.astype(int))
        weights      = 1.0 / class_counts[y_tr.astype(int)]
        sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

        tr_dl  = DataLoader(tr_ds,  batch_size=TAB_BATCH_SIZE, sampler=sampler,
                            num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=TAB_BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

        criterion = FocalLoss(FOCAL_GAMMA, make_pos_weight(y_tr), LABEL_SMOOTH)
        model     = TabularOnlyModel(INPUT_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)

        print(f"  Trainable: {model.count_trainable()/1e3:.1f}K  "
              f"Tab dim: {INPUT_DIM}")

        best_auc   = -np.inf
        best_state = None
        best_probs = None
        history    = {'loss': [], 'val_auc': []}

        ep_pbar = tqdm(range(1, N_EPOCHS + 1), desc=f"  Fold{fold}",
                       bar_format='{l_bar}{bar:30}{r_bar}')
        for ep in ep_pbar:
            loss = train_epoch_tabular(model, tr_dl, optimizer, criterion, DEVICE,
                                       desc=f"Ep{ep:>3}")
            scheduler.step()
            probs_v, lbls_v = eval_epoch_tabular(model, val_dl, DEVICE)
            try:
                auc = roc_auc_score(lbls_v, probs_v)
            except Exception:
                auc = 0.5

            history['loss'].append(loss)
            history['val_auc'].append(auc)

            if auc > best_auc:
                best_auc   = auc
                best_probs = probs_v.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_torch_atomic(
                    Path(output_dir) / f'fold{fold}_tabular_only_best_ckpt.pth', {
                    'epoch':      ep,
                    'fold':       fold,
                    'model_state': best_state,
                    'val_auc':    auc,
                    'input_dim':  INPUT_DIM,
                    'val_sample_indices': [int(index) for index in val_idx],
                    'val_labels': y_val.tolist(),
                    'preprocessors': preproc,
                })

            ep_pbar.set_postfix({'loss': f'{loss:.4f}', 'auc': f'{auc:.4f}',
                                 'best': f'{best_auc:.4f}'})

        oof_probs[val_idx] = best_probs
        fold_aucs.append(best_auc)
        fold_history.append(history)
        print(f"  Fold {fold} best AUC = {best_auc:.4f}")

        del model, optimizer
        torch.cuda.empty_cache()

    return oof_probs, fold_aucs, fold_history


def train_multimodal(y, names, X_raw, img_paths, fold_splits, output_dir):
    print(f"\n{'='*60}")
    print("MODE: multimodal")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    oof_probs    = np.zeros(len(y))
    fold_aucs    = []
    fold_history = []
    gate_weights = []

    for fold, sp in enumerate(fold_splits, 1):
        tr_idx  = sp['train']
        val_idx = sp['val']
        print(f"\n-- Fold {fold} --")

        y_tr  = y[tr_idx]
        y_val = y[val_idx]
        n_tr  = [names[i] for i in tr_idx]
        n_val = [names[i] for i in val_idx]
        ip_tr  = [img_paths[i] for i in tr_idx]
        ip_val = [img_paths[i] for i in val_idx]

        # Fit preprocessors on training fold only
        X_tr_raw  = X_raw.iloc[tr_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        X_tr_sc, preproc = fit_fold_preprocessors(X_tr_raw)
        X_val_sc         = transform_fold(X_val_raw, preproc)
        INPUT_DIM        = X_tr_sc.shape[1]

        tr_ds  = MultimodalDataset(X_tr_sc,  n_tr,  y_tr,  ip_tr,  TRAIN_TRANSFORM)
        val_ds = MultimodalDataset(X_val_sc, n_val, y_val, ip_val, VAL_TRANSFORM)

        class_counts = np.bincount(y_tr.astype(int))
        weights      = 1.0 / class_counts[y_tr.astype(int)]
        sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

        _bs   = min(IMG_BATCH_SIZE, max(1, len(n_tr) // 2))
        _drop = len(n_tr) >= IMG_BATCH_SIZE * 2
        tr_dl  = DataLoader(tr_ds,  batch_size=_bs, sampler=sampler,
                            num_workers=4, collate_fn=multimodal_collate_fn,
                            pin_memory=True, drop_last=_drop)
        val_dl = DataLoader(val_ds, batch_size=max(1, IMG_BATCH_SIZE // 2),
                            shuffle=False, num_workers=4,
                            collate_fn=multimodal_collate_fn, pin_memory=True)

        criterion = FocalLoss(FOCAL_GAMMA, make_pos_weight(y_tr), LABEL_SMOOTH)
        encoder   = load_biomedclip_encoder(DEVICE)
        model     = MultimodalModel(encoder, INPUT_DIM).to(DEVICE)

        bb_lr     = LR * BACKBONE_LR_RATIO
        optimizer = torch.optim.AdamW([
            {'params': model.get_backbone_params(), 'lr': bb_lr,  'weight_decay': 0.1},
            {'params': model.get_head_params(),     'lr': LR,     'weight_decay': 1e-2},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS, eta_min=LR * 0.005)

        bb_p   = sum(p.numel() for p in model.get_backbone_params())
        head_p = sum(p.numel() for p in model.get_head_params())
        print(f"  Trainable: backbone {bb_p/1e6:.2f}M @ lr={bb_lr:.2e}, "
              f"head {head_p/1e3:.1f}K @ lr={LR:.2e}")

        best_auc   = -np.inf
        best_state = None
        best_probs = None
        history    = {'loss': [], 'val_auc': []}

        ep_pbar = tqdm(range(1, N_EPOCHS + 1), desc=f"  Fold{fold}",
                       bar_format='{l_bar}{bar:30}{r_bar}')
        for ep in ep_pbar:
            loss = train_epoch_multimodal(model, tr_dl, optimizer, criterion, DEVICE,
                                          desc=f"Ep{ep:>3}")
            scheduler.step()
            probs_v, lbls_v = eval_epoch_multimodal(model, val_dl, DEVICE)
            try:
                auc = roc_auc_score(lbls_v, probs_v)
            except Exception:
                auc = 0.5

            history['loss'].append(loss)
            history['val_auc'].append(auc)

            if auc > best_auc:
                best_auc   = auc
                best_probs = probs_v.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_torch_atomic(
                    Path(output_dir) / f'fold{fold}_multimodal_best_ckpt.pth', {
                    'epoch':         ep,
                    'fold':          fold,
                    'model_state':   best_state,
                    'val_auc':       auc,
                    'feat_dim':      FEAT_DIM,
                    'img_out_dim':   256,
                    'tab_input_dim': INPUT_DIM,
                    'val_sample_indices': [int(index) for index in val_idx],
                    'val_labels':    y_val.tolist(),
                    'preprocessors': preproc,
                })

            ep_pbar.set_postfix({'loss': f'{loss:.4f}', 'auc': f'{auc:.4f}',
                                 'best': f'{best_auc:.4f}'})

        # Record gate weights using best model
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        model.eval()
        gate_vals = []
        with torch.no_grad():
            for tab_b, imgs_b, masks_b, _ in val_dl:
                tab_b, imgs_b, masks_b = (tab_b.to(DEVICE), imgs_b.to(DEVICE),
                                          masks_b.to(DEVICE))
                img_feat, _ = model.img_branch(imgs_b, masks_b)
                tab_feat    = model.tab_branch(tab_b)
                gate = model.gate_fusion.gate_net(
                    torch.cat([img_feat, tab_feat], dim=1))
                gate_vals.append(gate.cpu().mean(dim=1).numpy())
        gate_mean = float(np.concatenate(gate_vals).mean())

        # Save deidentified attention weights.
        save_attention_weights_image(
            model, val_dl, val_idx, DEVICE,
            save_dir=Path(output_dir) / 'attention_weights',
            fold=fold, val_probs=best_probs)

        oof_probs[val_idx] = best_probs
        gate_weights.append(gate_mean)
        fold_aucs.append(best_auc)
        fold_history.append(history)
        print(f"  Gate mean: {gate_mean:.3f}  Fold {fold} best AUC = {best_auc:.4f}")

        del model, encoder, optimizer
        torch.cuda.empty_cache()

    return oof_probs, fold_aucs, fold_history, gate_weights


# Main workflow
def main():
    global EXCEL_PATH, IMAGE_DIR, OUTPUT_ROOT, FOLD_SPLIT_PATH, MODEL_MODE

    EXCEL_PATH = prompt_for_data_path()
    IMAGE_DIR = prompt_for_image_directory()
    MODEL_MODE = prompt_for_model_mode()
    OUTPUT_ROOT = Path(__file__).resolve().parent / 'results_deep_learning_invasion_binary'
    FOLD_SPLIT_PATH = OUTPUT_ROOT / 'fold_splits_shared.json'
    output_dir = OUTPUT_ROOT / MODE_NAMES[MODEL_MODE]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"Device: {DEVICE}")
    print(f"Mode: {MODEL_MODE}")
    print("Input workbook: provided")
    print("Image directory: provided")
    print(f"Output directory: {OUTPUT_ROOT.name}/{output_dir.name}")

    df_all      = load_and_validate_dataset(EXCEL_PATH)
    img_col     = df_all.iloc[:, IMG_COL_IDX]
    layer_col   = df_all.iloc[:, LAYER_COL_IDX]
    layer_clean = layer_col.astype(str).str.strip()

    mask = (img_col == 1) & layer_clean.isin(['黏膜层', '黏膜下层'])
    df   = df_all[mask].copy().reset_index(drop=True)
    y    = (df.iloc[:, LAYER_COL_IDX].astype(str).str.strip()
              .map({'黏膜层': 0, '黏膜下层': 1}).values)
    names = df.iloc[:, NAME_COL_IDX].tolist()

    if df.empty:
        raise ValueError("No eligible T1a/T1b invasion-depth samples were found.")

    print(f"Valid samples before image check: {len(df)}")
    print(f"  T1a (mucosal): {(y == 0).sum()}")
    print(f"  T1b (submucosal): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

    # -- Filter out patients with no actual images --
    # Done before fold split so all three modes use the identical patient set.
    all_img_paths = [get_image_paths(n, IMAGE_DIR) for n in names]
    has_images    = [len(p) > 0 for p in all_img_paths]
    n_no_img      = sum(1 for h in has_images if not h)
    if n_no_img > 0:
        print(f"Samples excluded because no image was found: {n_no_img}")
        keep      = [i for i, h in enumerate(has_images) if h]
        df        = df.iloc[keep].reset_index(drop=True)
        y         = y[keep]
        names     = [names[i] for i in keep]
        all_img_paths = [all_img_paths[i] for i in keep]
        print(f"Eligible samples after image check: {len(y)}")

    if len(y) < N_FOLDS or len(np.unique(y)) < 2:
        raise ValueError("The eligible cohort is insufficient for stratified cross-validation.")

    img_paths = all_img_paths
    n_imgs    = [len(p) for p in img_paths]
    print(f"Images per sample: min={min(n_imgs)} max={max(n_imgs)} avg={np.mean(n_imgs):.1f}")

    run_signature = build_run_signature(y, img_paths)
    if FOLD_SPLIT_PATH.exists():
        with FOLD_SPLIT_PATH.open('r', encoding='utf-8') as source:
            split_data = json.load(source)
        fold_splits = split_data.get('splits')
        split_is_valid = (
            split_data.get('run_signature') == run_signature
            and split_data.get('labels') == y.tolist()
            and split_data.get('n_folds') == N_FOLDS
            and isinstance(fold_splits, list)
            and len(fold_splits) == N_FOLDS
        )
        if not split_is_valid:
            print(
                "[FATAL] Existing fold splits do not match the current data, "
                "code, images, or configuration. Move or delete the existing "
                "results directory before starting again."
            )
            raise SystemExit(1) from None
        print(f"Shared fold splits loaded: {N_FOLDS} folds")
    else:
        cv          = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_splits = [{'train': tr.tolist(), 'val': va.tolist()}
                       for tr, va in cv.split(np.zeros(len(y)), y)]
        save_json_atomic(FOLD_SPLIT_PATH, {
            'run_signature': run_signature,
            'labels': y.tolist(),
            'n_folds': N_FOLDS,
            'splits': fold_splits,
        })
        print(f"Shared fold splits created: {N_FOLDS} folds")

    # -- Raw tabular features (no preprocessing yet) --
    X_raw = build_raw_tabular(df)

    # -- Train selected mode --
    gate_weights = []

    if MODEL_MODE == 'image-only':
        oof_probs, fold_aucs, _ = train_image_only(
            y, names, img_paths, fold_splits, output_dir)

    elif MODEL_MODE == 'tabular-only':
        oof_probs, fold_aucs, _ = train_tabular_only(
            y, names, X_raw, fold_splits, output_dir)

    else:
        oof_probs, fold_aucs, _, gate_weights = train_multimodal(
            y, names, X_raw, img_paths, fold_splits, output_dir)

    metrics = compute_metrics(y, oof_probs, fold_aucs)

    print(f"\n{'='*60}")
    print(f"  Mode          : {MODEL_MODE}")
    print(f"  OOF AUC       : {metrics['oof_auc']:.4f}")
    print(f"  Mean Fold AUC : {metrics['mean_fold_auc']:.4f} ± {metrics['std_fold_auc']:.4f}")
    print(f"  AP             : {metrics['pr_auc']:.4f}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Balanced Acc.  : {metrics['balanced_accuracy']:.4f}")
    print(f"  F1             : {metrics['f1']:.4f}")
    print(f"  Sensitivity   : {metrics['sensitivity']:.4f}")
    print(f"  Specificity   : {metrics['specificity']:.4f}")
    print(f"  Threshold      : {PREDICTION_THRESHOLD:.3f}")
    if gate_weights:
        print(f"  Gate means    : {[f'{g:.3f}' for g in gate_weights]}")
    print(f"{'='*60}")

    predictions_path = output_dir / 'predictions.csv'
    pd.DataFrame({
        'sample_index': np.arange(len(y), dtype=int),
        'true_label': y.astype(int),
        'predicted_probability': oof_probs,
        'prediction_at_0_5': metrics['oof_preds'].astype(int),
        'correct': (metrics['oof_preds'] == y).astype(int),
    }).to_csv(predictions_path, index=False, encoding='utf-8-sig')

    fold_metrics_path = output_dir / 'fold_metrics.csv'
    pd.DataFrame({
        'fold': np.arange(1, len(fold_aucs) + 1, dtype=int),
        'auroc': np.asarray(fold_aucs, dtype=float),
        'gate_mean': (
            np.asarray(gate_weights, dtype=float)
            if gate_weights else np.full(len(fold_aucs), np.nan)
        ),
    }).to_csv(fold_metrics_path, index=False, encoding='utf-8-sig')

    mean_roc_path = output_dir / 'mean_roc.csv'
    build_mean_roc_table(y, oof_probs, fold_splits).to_csv(
        mean_roc_path,
        index=False,
        encoding='utf-8-sig',
    )

    metrics_out = {
        'mode': MODEL_MODE,
        'n_samples': int(len(y)),
        'n_positive': int((y == 1).sum()),
        'n_negative': int((y == 0).sum()),
        'n_folds': N_FOLDS,
        'n_epochs': N_EPOCHS,
        'threshold': PREDICTION_THRESHOLD,
        'oof_auc': float(metrics['oof_auc']),
        'mean_fold_auc': float(metrics['mean_fold_auc']),
        'std_fold_auc': float(metrics['std_fold_auc']),
        'average_precision': float(metrics['pr_auc']),
        'accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'f1': float(metrics['f1']),
        'sensitivity': float(metrics['sensitivity']),
        'specificity': float(metrics['specificity']),
        'confusion_matrix': metrics['cm'].astype(int).tolist(),
        'fold_aucs': [float(value) for value in fold_aucs],
        'gate_means': [float(value) for value in gate_weights],
        'roc_false_positive_rate': metrics['fpr'].tolist(),
        'roc_true_positive_rate': metrics['tpr'].tolist(),
    }
    metrics_path = output_dir / 'metrics.json'
    save_json_atomic(metrics_path, metrics_out)

    print(f"[Saved] {OUTPUT_ROOT.name}/{output_dir.name}/{predictions_path.name}")
    print(f"[Saved] {OUTPUT_ROOT.name}/{output_dir.name}/{fold_metrics_path.name}")
    print(f"[Saved] {OUTPUT_ROOT.name}/{output_dir.name}/{mean_roc_path.name}")
    print(f"[Saved] {OUTPUT_ROOT.name}/{output_dir.name}/{metrics_path.name}")
    print("Training complete.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Stopped] Run interrupted by user.")
        raise SystemExit(130) from None
    except Exception as error:
        print(f"[FATAL] Run failed: {type(error).__name__}")
        raise SystemExit(1) from None

