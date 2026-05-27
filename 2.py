pretrain_available = {}
for _, row in df[df['train'] == True].iterrows():
    path = os.path.join(TRAIN_PATH, row['attachment_id'] + '.mp4')
    if os.path.exists(path):
        pretrain_available[row['text']] = pretrain_available.get(row['text'], 0) + 1

total_videos = sum(pretrain_available.values())
total_classes = len(pretrain_available)
classes_5plus = sum(1 for v in pretrain_available.values() if v >= 5)
classes_10plus = sum(1 for v in pretrain_available.values() if v >= 10)

print(f"Всего видео для предобучения: {total_videos}")
print(f"Всего классов: {total_classes}")
print(f"Классов с 5+ видео: {classes_5plus}")
print(f"Классов с 10+ видео: {classes_10plus}")
print(f"\nРаспределение (топ-30 по количеству):")
for g, c in sorted(pretrain_available.items(), key=lambda x: -x[1])[:30]:
    print(f"  {g:30s}: {c}")
print(f"\nРаспределение (худшие 20):")
for g, c in sorted(pretrain_available.items(), key=lambda x: x[1])[:20]:
    print(f"  {g:30s}: {c}")

import os, math, urllib.request
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from google.colab import drive
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

drive.mount('/content/drive')

class Config:
    SEQUENCE_LEN       = 40
    NUM_HANDS          = 2
    LANDMARKS_PER_HAND = 21
    COORDS_PER_LM      = 3                            # x, y, z
    ANGLES_PER_HAND    = 15                           # углы между суставами

    COORD_FEATURES  = 126   # 21 * 3 * 2
    ANGLE_FEATURES  = 30    # 15 * 2
    RAW_FEATURES    = 156
    TOTAL_FEATURES  = 312

    BATCH_SIZE    = 32
    EPOCHS        = 100
    LR            = 1e-3
    LR_FINETUNE   = 1e-4
    DROPOUT       = 0.4

    NUM_FOLDS     = 5
    SEED          = 42

    TOP_K         = 20
    PRETRAIN_MIN_SAMPLES = 5

    # Пути
    DRIVE_ROOT    = '/content/drive/MyDrive'
    DATASET_PATH  = '/content/drive/MyDrive/slovo_full'

cfg = Config()

TRAIN_PATH      = os.path.join(cfg.DATASET_PATH, 'train')
TEST_PATH       = os.path.join(cfg.DATASET_PATH, 'test')
ANNOTATION_PATH = os.path.join(cfg.DATASET_PATH, 'annotations.csv')

!pip install mediapipe -q
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_TASK_PATH = 'hand_landmarker.task'
if not os.path.exists(_TASK_PATH):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/1/hand_landmarker.task',
        _TASK_PATH
    )

_base    = mp_python.BaseOptions(model_asset_path=_TASK_PATH)
_options = mp_vision.HandLandmarkerOptions(
    base_options=_base,
    num_hands=cfg.NUM_HANDS,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = mp_vision.HandLandmarker.create_from_options(_options)


ANGLE_TRIPLETS = [
    (1, 2, 3),   (2, 3, 4),             # большой палец
    (5, 6, 7),   (6, 7, 8),   (0, 5, 9),# указательный
    (9, 10, 11), (10, 11, 12),(0, 9, 13),# средний
    (13, 14, 15),(14, 15, 16),(0, 13, 17),# безымянный
    (17, 18, 19),(18, 19, 20),(0, 17, 5), # мизинец
    (5, 9, 13),                           # ладонь (поперёк)
]
assert len(ANGLE_TRIPLETS) == cfg.ANGLES_PER_HAND


def _angle_between(a, b, c):
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))


def compute_hand_angles(hand_xyz: np.ndarray) -> np.ndarray:
    return np.array([_angle_between(hand_xyz[a], hand_xyz[b], hand_xyz[c])
                     for a, b, c in ANGLE_TRIPLETS], dtype=np.float32)


def normalize_hand(pts: np.ndarray) -> np.ndarray:

    centered = pts - pts[0]
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 1e-6:
        centered = centered / scale
    return centered


def extract_frame_features(res) -> np.ndarray:

    vec = np.zeros(cfg.RAW_FEATURES, dtype=np.float32)
    if not res.hand_landmarks:
        return vec

    for h_idx, hand in enumerate(res.hand_landmarks[:cfg.NUM_HANDS]):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand[:21]], dtype=np.float32)
        pts_norm = normalize_hand(pts)

        base_coord  = h_idx * (cfg.LANDMARKS_PER_HAND * cfg.COORDS_PER_LM + cfg.ANGLES_PER_HAND)
        base_angle  = base_coord + cfg.LANDMARKS_PER_HAND * cfg.COORDS_PER_LM

        vec[base_coord : base_coord + 63] = pts_norm.flatten()
        vec[base_angle : base_angle + cfg.ANGLES_PER_HAND] = compute_hand_angles(pts_norm)

    return vec


def add_velocity(seq: np.ndarray) -> np.ndarray:

    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=1)


def extract_sequence(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < cfg.SEQUENCE_LEN:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, cfg.SEQUENCE_LEN, dtype=int)
    seq = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            seq.append(np.zeros(cfg.RAW_FEATURES, dtype=np.float32))
            continue

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res  = detector.detect(img)
        seq.append(extract_frame_features(res))

    cap.release()
    while len(seq) < cfg.SEQUENCE_LEN:
        seq.append(np.zeros(cfg.RAW_FEATURES, dtype=np.float32))

    seq_arr = np.array(seq[:cfg.SEQUENCE_LEN], dtype=np.float32)
    return add_velocity(seq_arr)   # (40, 312)



def augment(seq: np.ndarray) -> list[np.ndarray]:
    out = [seq]

    for sigma in [0.005, 0.012]:
        out.append(seq + np.random.normal(0, sigma, seq.shape).astype(np.float32))

    shift = np.random.randint(-4, 5)
    if shift != 0:
        shifted = np.roll(seq, shift, axis=0)
        if shift > 0:
            shifted[:shift] = seq[0]
        else:
            shifted[shift:] = seq[-1]
        out.append(shifted)

    out.append((seq * np.random.uniform(0.88, 1.12)).astype(np.float32))

    mirrored = seq.copy()
    for h in range(cfg.NUM_HANDS):
        base = h * (cfg.LANDMARKS_PER_HAND * cfg.COORDS_PER_LM + cfg.ANGLES_PER_HAND)
        for i in range(cfg.LANDMARKS_PER_HAND):
            xi = base + i * 3
            mirrored[:, xi] = 1.0 - seq[:, xi]

            xi_vel = xi + cfg.RAW_FEATURES
            mirrored[:, xi_vel] = -seq[:, xi_vel]
    out.append(mirrored)

    return out



def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(seq_len)[:, None]
    dims      = np.arange(d_model)[None, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[None, ...], dtype=tf.float32)   # (1, T, d_model)


def transformer_encoder_block(x, num_heads: int, ff_dim: int, dropout: float):
    d_model = x.shape[-1]
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x)
    attn_out = tf.keras.layers.Dropout(dropout)(attn_out)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = tf.keras.layers.Dense(ff_dim, activation='gelu')(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(d_model)(ff)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    x  = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_transformer_model(num_classes: int,
                             d_model: int = 128,
                             num_heads: int = 4,
                             num_layers: int = 3,
                             ff_dim: int = 256,
                             dropout: float = cfg.DROPOUT) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(cfg.SEQUENCE_LEN, cfg.TOTAL_FEATURES))

    x = tf.keras.layers.Dense(d_model)(inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    pe = positional_encoding(cfg.SEQUENCE_LEN, d_model)
    x  = x + pe

    x = tf.keras.layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)



def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def _loss(y_true, y_pred):
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce     = -y_true * tf.math.log(y_pred)
        w      = alpha * y_true * (1 - y_pred) ** gamma
        return tf.reduce_mean(tf.reduce_sum(w * ce, axis=1))
    return _loss


def get_callbacks(monitor: str = 'val_accuracy'):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=15,
            restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=6, min_lr=1e-7
        ),
    ]


print("Читаем аннотации...")
df = pd.read_csv(ANNOTATION_PATH, sep='\t')
df = df[df['text'] != 'no_event']

counts = df['text'].value_counts()
top20  = counts.head(cfg.TOP_K).index.tolist()

pretrain_classes = counts[counts >= cfg.PRETRAIN_MIN_SAMPLES].index.tolist()
print(f"Топ-{cfg.TOP_K} финальных классов: {top20}")
print(f"Классов для предобучения: {len(pretrain_classes)}")
print(f"Распределение топ-{cfg.TOP_K}:\n{counts.head(cfg.TOP_K)}")

total_top20 = sum(counts[l] for l in top20)
class_weights_top20 = {
    i: total_top20 / (cfg.TOP_K * counts[top20[i]])
    for i in range(cfg.TOP_K)
}

train_df_top20    = df[(df['train'] == True)  & (df['text'].isin(top20))].reset_index(drop=True)
test_df_top20     = df[(df['train'] == False) & (df['text'].isin(top20))].reset_index(drop=True)
train_df_pretrain = df[(df['train'] == True)  & (df['text'].isin(pretrain_classes))].reset_index(drop=True)

print(f"\nТоп-20  → Train: {len(train_df_top20)}, Test: {len(test_df_top20)}")
print(f"Предобучение → Train: {len(train_df_pretrain)} ({len(pretrain_classes)} классов)")



def extract_split(split_df: pd.DataFrame,
                  folder: str,
                  desc: str = '') -> tuple[np.ndarray, list[str]]:
    X, y = [], []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
        path = os.path.join(folder, row['attachment_id'] + '.mp4')
        if not os.path.exists(path):
            continue
        seq = extract_sequence(path)
        if seq is not None:
            X.append(seq)
            y.append(row['text'])
    return np.array(X, dtype=np.float32), np.array(y)


print("\n=== Извлечение признаков: предобучение ===")
X_pre, y_pre = extract_split(train_df_pretrain, TRAIN_PATH, 'Pretrain')
print(f"Претрейн: {X_pre.shape}")

print("\n=== Извлечение признаков: топ-20 (train) ===")
X_train, y_train_str = extract_split(train_df_top20, TRAIN_PATH, 'Train-top20')
print("\n=== Извлечение признаков: топ-20 (test) ===")
X_test,  y_test_str  = extract_split(test_df_top20,  TEST_PATH,  'Test-top20')
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

le_pre  = LabelEncoder().fit(y_pre)
le_top  = LabelEncoder().fit(top20)

y_pre_enc   = le_pre.transform(y_pre)
y_train_enc = le_top.transform(y_train_str)
y_test_enc  = le_top.transform(y_test_str)

NUM_PRETRAIN = len(le_pre.classes_)
print(f"Классов для предобучения: {NUM_PRETRAIN}")

import os, urllib.request
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

drive.mount('/content/drive')

class Config:
    SEQUENCE_LEN       = 40
    NUM_HANDS          = 2
    LANDMARKS_PER_HAND = 21
    COORDS_PER_LM      = 3
    ANGLES_PER_HAND    = 15
    COORD_FEATURES     = 126
    ANGLE_FEATURES     = 30
    RAW_FEATURES       = 156
    TOTAL_FEATURES     = 312

    BATCH_SIZE         = 32
    EPOCHS             = 100
    LR                 = 1e-3
    LR_FINETUNE        = 5e-5
    DROPOUT            = 0.4
    NUM_FOLDS          = 5
    SEED               = 42
    TOP_K              = 20
    DRIVE_ROOT         = '/content/drive/MyDrive'

cfg = Config()

print("Загружаем данные с диска...")
X_pre       = np.load(f'{cfg.DRIVE_ROOT}/X_pre.npy')
y_pre       = np.load(f'{cfg.DRIVE_ROOT}/y_pre.npy',       allow_pickle=True)
X_train     = np.load(f'{cfg.DRIVE_ROOT}/X_train.npy')
y_train_str = np.load(f'{cfg.DRIVE_ROOT}/y_train_str.npy', allow_pickle=True)
X_test      = np.load(f'{cfg.DRIVE_ROOT}/X_test.npy')
y_test_str  = np.load(f'{cfg.DRIVE_ROOT}/y_test_str.npy',  allow_pickle=True)

print(f"X_pre:   {X_pre.shape}")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

le_pre = LabelEncoder().fit(y_pre)
top20  = sorted(set(y_train_str))
le_top = LabelEncoder().fit(top20)

y_pre_enc   = le_pre.transform(y_pre)
y_train_enc = le_top.transform(y_train_str)
y_test_enc  = le_top.transform(y_test_str)

train_counts = Counter(y_train_str)
total_train  = len(y_train_str)
class_weights = {
    le_top.transform([cls])[0]: total_train / (cfg.TOP_K * cnt)
    for cls, cnt in train_counts.items()
}

NUM_PRETRAIN = len(le_pre.classes_)
print(f"Классов предобучения: {NUM_PRETRAIN}")
print(f"Топ-20 классов: {top20}")


def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, None]
    dims      = np.arange(d_model)[None, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[None, ...], dtype=tf.float32)

def transformer_encoder_block(x, num_heads, ff_dim, dropout):
    d_model  = x.shape[-1]
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x)
    attn_out = tf.keras.layers.Dropout(dropout)(attn_out)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = tf.keras.layers.Dense(ff_dim, activation='gelu')(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(d_model)(ff)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    x  = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x

def build_transformer_model(num_classes, d_model=128, num_heads=4,
                             num_layers=3, ff_dim=256, dropout=cfg.DROPOUT):
    inputs = tf.keras.Input(shape=(cfg.SEQUENCE_LEN, cfg.TOTAL_FEATURES))
    x = tf.keras.layers.Dense(d_model)(inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = x + positional_encoding(cfg.SEQUENCE_LEN, d_model)
    x = tf.keras.layers.Dropout(dropout)(x)
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def focal_loss(gamma=2.0, alpha=0.25):
    def _loss(y_true, y_pred):
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce     = -y_true * tf.math.log(y_pred)
        w      = alpha * y_true * (1 - y_pred) ** gamma
        return tf.reduce_mean(tf.reduce_sum(w * ce, axis=1))
    return _loss

def augment(seq):
    out = [seq]
    for sigma in [0.005, 0.012]:
        out.append(seq + np.random.normal(0, sigma, seq.shape).astype(np.float32))
    shift = np.random.randint(-4, 5)
    if shift != 0:
        shifted = np.roll(seq, shift, axis=0)
        if shift > 0: shifted[:shift] = seq[0]
        else:         shifted[shift:] = seq[-1]
        out.append(shifted)
    out.append((seq * np.random.uniform(0.88, 1.12)).astype(np.float32))
    mirrored = seq.copy()
    coords_per_hand = cfg.LANDMARKS_PER_HAND * cfg.COORDS_PER_LM + cfg.ANGLES_PER_HAND
    for h in range(cfg.NUM_HANDS):
        base = h * coords_per_hand
        for i in range(cfg.LANDMARKS_PER_HAND):
            xi = base + i * 3
            mirrored[:, xi]                    =  1.0 - seq[:, xi]
            mirrored[:, xi + cfg.RAW_FEATURES] = -seq[:, xi + cfg.RAW_FEATURES]
    out.append(mirrored)
    return out

print("\n=== ФАЗА 1: Предобучение на всех классах ===")

unique, counts_arr = np.unique(y_pre_enc, return_counts=True)
valid_mask = np.isin(y_pre_enc, unique[counts_arr >= 2])
X_pre_f    = X_pre[valid_mask]
y_pre_f    = y_pre_enc[valid_mask]

le_pre2   = LabelEncoder().fit(y_pre_f)
y_pre_f2  = le_pre2.transform(y_pre_f)
NUM_PRE2  = len(le_pre2.classes_)
print(f"После фильтрации: {X_pre_f.shape[0]} видео, {NUM_PRE2} классов")

y_pre_cat = tf.keras.utils.to_categorical(y_pre_f2, NUM_PRE2)
X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(
    X_pre_f, y_pre_cat, test_size=0.15, random_state=cfg.SEED
)
print(f"Pretrain — train: {X_pre_tr.shape[0]}, val: {X_pre_val.shape[0]}")

tf.keras.backend.clear_session()
pretrain_model = build_transformer_model(num_classes=NUM_PRE2)
pretrain_model.compile(
    optimizer=tf.keras.optimizers.Adam(cfg.LR),
    loss=focal_loss(2.0, 0.25),
    metrics=['accuracy']
)

PRETRAIN_CKPT = f'{cfg.DRIVE_ROOT}/slovo_pretrained_best.keras'

pretrain_model.fit(
    X_pre_tr, y_pre_tr,
    validation_data=(X_pre_val, y_pre_val),
    epochs=120,                          # ← было 60, теперь 120
    batch_size=cfg.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=8, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(  # ← сохраняем лучшую эпоху
            PRETRAIN_CKPT,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
    ],
    verbose=1
)

pretrain_model = tf.keras.models.load_model(PRETRAIN_CKPT, compile=False)
pretrain_model.save(f'{cfg.DRIVE_ROOT}/slovo_pretrained.keras')
print("Лучшая предобученная модель сохранена.")


def make_finetune_model(pretrained, num_classes):
    feature_output = pretrained.layers[-3].output
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='new_head'
    )(feature_output)
    return tf.keras.Model(pretrained.input, outputs)

print("\n=== ФАЗА 2: Fine-tune (5-Fold CV) на топ-20 ===")

skf = StratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)
fold_models   = []
fold_val_accs = []
best_val_acc  = 0.0
best_model    = None

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):
    print(f"\n{'─'*55}")
    print(f"  FOLD {fold+1}/{cfg.NUM_FOLDS}")

    X_tr,  y_tr  = X_train[tr_idx], y_train_enc[tr_idx]
    X_val, y_val = X_train[val_idx], y_train_enc[val_idx]

    X_aug, y_aug = [], []
    for seq, label in zip(X_tr, y_tr):
        for aug_seq in augment(seq):
            X_aug.append(aug_seq)
            y_aug.append(label)
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    print(f"  После аугментации: {len(X_aug)} samples")

    y_aug_cat = tf.keras.utils.to_categorical(y_aug, cfg.TOP_K)
    y_val_cat = tf.keras.utils.to_categorical(y_val, cfg.TOP_K)

    FOLD_CKPT = f'{cfg.DRIVE_ROOT}/fold_{fold+1}_best.keras'

    tf.keras.backend.clear_session()
    pretrained_reload = tf.keras.models.load_model(
        f'{cfg.DRIVE_ROOT}/slovo_pretrained.keras', compile=False
    )
    model = make_finetune_model(pretrained_reload, num_classes=cfg.TOP_K)

    for layer in model.layers[:-3]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LR),
        loss=focal_loss(2.0, 0.25),
        metrics=['accuracy']
    )
    model.fit(
        X_aug, y_aug_cat,
        validation_data=(X_val, y_val_cat),
        epochs=20,
        batch_size=cfg.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True, mode='max'
            ),
        ],
        verbose=0
    )

    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LR_FINETUNE),
        loss=focal_loss(2.0, 0.25),
        metrics=['accuracy']
    )
    model.fit(
        X_aug, y_aug_cat,
        validation_data=(X_val, y_val_cat),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=6, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                FOLD_CKPT,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
            ),
        ],
        class_weight=class_weights,
        verbose=1
    )

    best_fold_model = tf.keras.models.load_model(FOLD_CKPT, compile=False)
    fold_models.append(best_fold_model)

    val_pred    = np.argmax(best_fold_model.predict(X_val, verbose=0), axis=1)
    val_bal_acc = balanced_accuracy_score(y_val, val_pred)
    val_std_acc = np.mean(val_pred == y_val)
    fold_val_accs.append(val_bal_acc)

    print(f"  Fold {fold+1} → Accuracy: {val_std_acc:.2%}, Balanced: {val_bal_acc:.2%}")

    if val_bal_acc > best_val_acc:
        best_val_acc = val_bal_acc
        best_model   = best_fold_model
        print(f"  ★ Новая лучшая модель (balanced acc = {best_val_acc:.2%})")

print(f"\nСредняя Balanced Val Acc: {np.mean(fold_val_accs):.2%} ± {np.std(fold_val_accs):.2%}")

print("\n=== ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ (ансамбль) ===")

ensemble_probs = np.mean(
    [m.predict(X_test, verbose=0) for m in fold_models], axis=0
)
y_pred_ens  = np.argmax(ensemble_probs, axis=1)
y_pred_best = np.argmax(best_model.predict(X_test, verbose=0), axis=1)

acc_best  = np.mean(y_pred_best == y_test_enc)
bacc_best = balanced_accuracy_score(y_test_enc, y_pred_best)
acc_ens   = np.mean(y_pred_ens  == y_test_enc)
bacc_ens  = balanced_accuracy_score(y_test_enc, y_pred_ens)

print(f"\nЛучшая одиночная модель:")
print(f"  Accuracy: {acc_best:.2%},  Balanced: {bacc_best:.2%}")
print(f"\nАнсамбль 5 моделей:")
print(f"  Accuracy: {acc_ens:.2%},   Balanced: {bacc_ens:.2%}")

unique_labels = np.unique(np.concatenate([y_test_enc, y_pred_ens]))
print("\nClassification Report (ансамбль):")
print(classification_report(
    y_test_enc, y_pred_ens,
    labels=unique_labels,
    target_names=[le_top.inverse_transform([i])[0] for i in unique_labels],
    zero_division=0
))


class_names = [le_top.inverse_transform([i])[0] for i in range(cfg.TOP_K)]

fig, axes = plt.subplots(1, 2, figsize=(28, 12))
for ax, preds, title in zip(
    axes,
    [y_pred_best, y_pred_ens],
    ['Best Single Model', 'Ensemble (5 folds)']
):
    present      = sorted(np.unique(np.concatenate([y_test_enc, preds])))
    present_names = [le_top.inverse_transform([i])[0] for i in present]
    cm = confusion_matrix(y_test_enc, preds, labels=present)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=present_names, yticklabels=present_names,
                cmap='Blues', ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(f'{cfg.DRIVE_ROOT}/confusion_matrix_v3.png', dpi=150)
plt.show()

errors = np.where(y_pred_ens != y_test_enc)[0]
print(f"\n=== АНАЛИЗ ОШИБОК (ансамбль) ===")
print(f"Всего ошибок: {len(errors)}/{len(y_test_enc)} ({len(errors)/len(y_test_enc):.1%})")
if len(errors):
    error_counts = Counter([(y_test_enc[i], y_pred_ens[i]) for i in errors])
    print("\nТоп-10 частых ошибок:")
    for (t, p), cnt in error_counts.most_common(10):
        print(f"  {class_names[t]:25s} → {class_names[p]:25s}: {cnt}×")

best_model.save(f'{cfg.DRIVE_ROOT}/slovo_best_v3.keras')
joblib.dump(le_top, f'{cfg.DRIVE_ROOT}/slovo_label_encoder.pkl')
joblib.dump(top20,  f'{cfg.DRIVE_ROOT}/slovo_top20.pkl')

print("\n" + "="*55)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print(f"  Классов:              {cfg.TOP_K}")
print(f"  Признаков на кадр:    {cfg.TOTAL_FEATURES}")
print(f"  Train samples:        {len(X_train)}")
print(f"  Test  samples:        {len(X_test)}")
print(f"  CV Balanced Acc:      {np.mean(fold_val_accs):.2%} ± {np.std(fold_val_accs):.2%}")
print(f"  Test Acc (best):      {acc_best:.2%}")
print(f"  Test Acc (ensemble):  {acc_ens:.2%}")
print(f"  Test Bal Acc (ens):   {bacc_ens:.2%}")
print("="*55)

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ANGLE_TRIPLETS = [
    (1,2,3),(2,3,4),(5,6,7),(6,7,8),(0,5,9),
    (9,10,11),(10,11,12),(0,9,13),(13,14,15),(14,15,16),
    (0,13,17),(17,18,19),(18,19,20),(0,17,5),(5,9,13),
]

_TASK_PATH = 'hand_landmarker.task'
if not os.path.exists(_TASK_PATH):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/1/hand_landmarker.task',
        _TASK_PATH
    )
_detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_TASK_PATH),
        num_hands=2
    )
)

def _angle_between(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))

def _extract_frame(res):
    vec = np.zeros(cfg.RAW_FEATURES, dtype=np.float32)
    if not res.hand_landmarks: return vec
    coords_per_hand = cfg.LANDMARKS_PER_HAND * cfg.COORDS_PER_LM + cfg.ANGLES_PER_HAND
    for h_idx, hand in enumerate(res.hand_landmarks[:2]):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand[:21]], dtype=np.float32)
        pts -= pts[0]
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 1e-6: pts /= scale
        base_c = h_idx * coords_per_hand
        base_a = base_c + 63
        vec[base_c:base_c+63] = pts.flatten()
        vec[base_a:base_a+15] = [_angle_between(pts[a], pts[b], pts[c])
                                  for a, b, c in ANGLE_TRIPLETS]
    return vec

def predict_video(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < cfg.SEQUENCE_LEN:
        cap.release()
        return "Видео слишком короткое", 0.0, {}

    seq = []
    for idx in np.linspace(0, total-1, cfg.SEQUENCE_LEN, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            seq.append(np.zeros(cfg.RAW_FEATURES, dtype=np.float32))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        seq.append(_extract_frame(_detector.detect(img)))
    cap.release()

    while len(seq) < cfg.SEQUENCE_LEN:
        seq.append(np.zeros(cfg.RAW_FEATURES, dtype=np.float32))

    seq_arr = np.array(seq[:cfg.SEQUENCE_LEN], dtype=np.float32)
    vel     = np.diff(seq_arr, axis=0, prepend=seq_arr[:1])
    seq_arr = np.concatenate([seq_arr, vel], axis=1)
    inp     = seq_arr.reshape(1, cfg.SEQUENCE_LEN, cfg.TOTAL_FEATURES)

    probs  = np.mean([m.predict(inp, verbose=0) for m in fold_models], axis=0)[0]
    top5   = np.argsort(probs)[::-1][:5]
    top5_d = {le_top.inverse_transform([i])[0]: float(probs[i]) for i in top5}
    label  = le_top.inverse_transform([top5[0]])[0]
    return label, float(probs[top5[0]]), top5_d

from google.colab import files
print("\nЗагрузите видео для проверки:")
uploaded = files.upload()
for fname in uploaded.keys():
    label, conf, top5 = predict_video(fname)
    print(f"\nВидео: {fname}")
    print(f"  Жест:        {label}")
    print(f"  Уверенность: {conf:.2%}")
    print("  Топ-5:")
    for g, p in top5.items():
        print(f"    {g:25s}: {p:.2%}")

