import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from google.colab import drive
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from collections import Counter


drive.mount('/content/drive')


class Config:
    SEQUENCE_LEN = 40
    NUM_HANDS = 2
    FEATURES_PER_HAND = 63
    TOTAL_FEATURES = FEATURES_PER_HAND * NUM_HANDS
    FEATURES_WITH_VELOCITY = TOTAL_FEATURES * 2

    BATCH_SIZE = 32
    EPOCHS = 80
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5

    NUM_FOLDS = 5
    RANDOM_SEED = 42


dataset_path = '/content/drive/MyDrive/slovo_full'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')
annotation_path = os.path.join(dataset_path, 'annotations.csv')

df = pd.read_csv(annotation_path, sep='\t')
df = df[df['text'] != 'no_event']


counts = df['text'].value_counts()
top20 = counts.head(20).index.tolist()
print(f"Топ-20 жестов: {top20}")
print(f"Распределение:\n{counts.head(20)}")


class_weights = {i: 1.0/counts[label] for i, label in enumerate(top20)}
total_samples = sum(counts[label] for label in top20)
class_weights = {i: (total_samples / (len(top20) * counts[label])) for i, label in enumerate(top20)}

train_df = df[(df['train'] == True) & (df['text'].isin(top20))].reset_index(drop=True)
test_df = df[(df['train'] == False) & (df['text'].isin(top20))].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")


!pip install mediapipe -q
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if not os.path.exists('hand_landmarker.task'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'hand_landmarker.task'
    )

base = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base,
    num_hands=Config.NUM_HANDS,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


def advanced_normalize(seq):
    normalized = []
    for frame in seq:
        points = frame.reshape(-1, 3)
        if np.all(points == 0):
            normalized.append(frame)
            continue

        center = np.mean(points[points[:, 0] > 0], axis=0) if np.any(points[:, 0] > 0) else np.mean(points, axis=0)
        centered = points - center

        scale = np.ptp(centered, axis=0).max()
        if scale > 1e-6:
            centered = centered / scale

        normalized.append(centered.flatten())
    return np.array(normalized, dtype=np.float32)


def extract_sequence_robust(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < Config.SEQUENCE_LEN:
        cap.release()
        return None

    indices = np.linspace(0, total-1, Config.SEQUENCE_LEN, dtype=int)
    seq = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(img)

        vec = [0.0] * Config.TOTAL_FEATURES
        if res.hand_landmarks:
            for hand_idx, hand in enumerate(res.hand_landmarks[:Config.NUM_HANDS]):
                base_idx = hand_idx * Config.FEATURES_PER_HAND
                for i, lm in enumerate(hand[:21]):
                    vec[base_idx + i*3] = lm.x
                    vec[base_idx + i*3+1] = lm.y
                    vec[base_idx + i*3+2] = lm.z
        seq.append(vec)

    cap.release()

    while len(seq) < Config.SEQUENCE_LEN:
        seq.append([0.0] * Config.TOTAL_FEATURES)

    seq_array = np.array(seq, dtype=np.float32)
    seq_normalized = advanced_normalize(seq_array)

    return seq_normalized


def advanced_augmentation(seq):
    augmented = [seq]


    for noise in [0.005, 0.01]:
        augmented.append(seq + np.random.normal(0, noise, seq.shape))


    if np.random.random() > 0.5:
        warp_factor = np.random.uniform(0.85, 1.15)
        indices = np.linspace(0, Config.SEQUENCE_LEN-1, Config.SEQUENCE_LEN)
        warped_indices = np.clip(indices * warp_factor, 0, Config.SEQUENCE_LEN-1).astype(int)
        augmented.append(seq[warped_indices])


    shift = np.random.randint(-3, 4)
    if shift != 0:
        shifted = np.roll(seq, shift, axis=0)
        if shift > 0:
            shifted[:shift] = seq[0]
        else:
            shifted[shift:] = seq[-1]
        augmented.append(shifted)


    if np.random.random() > 0.5:
        scale_factor = np.random.uniform(0.85, 1.15)
        augmented.append(seq * scale_factor)

    return augmented


def create_model(num_classes):
    inputs = tf.keras.Input(shape=(Config.SEQUENCE_LEN, Config.FEATURES_WITH_VELOCITY))

    x = tf.keras.layers.BatchNormalization()(inputs)

    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)


    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    context = tf.keras.layers.Dot(axes=1)([x, attention])

    x = tf.keras.layers.Dense(128, activation='relu')(context)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * (1 - y_pred)**gamma
        focal = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal, axis=1))
    return focal_loss_fixed


print("Извлечение признаков...")
X_data = []
y_data = []
video_ids = []

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    path = os.path.join(train_path, row['attachment_id'] + '.mp4')
    if os.path.exists(path):
        seq = extract_sequence_robust(path)
        if seq is not None:
            X_data.append(seq)
            y_data.append(row['text'])
            video_ids.append(('train', row['attachment_id']))

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    path = os.path.join(test_path, row['attachment_id'] + '.mp4')
    if os.path.exists(path):
        seq = extract_sequence_robust(path)
        if seq is not None:
            X_data.append(seq)
            y_data.append(row['text'])
            video_ids.append(('test', row['attachment_id']))

X_data = np.array(X_data)
y_data = np.array(y_data)
print(f"Total samples: {X_data.shape}")


def add_velocity_features(seq):
    velocity = np.diff(seq, axis=0)
    velocity = np.pad(velocity, ((0,1), (0,0)), 'constant')
    return np.concatenate([seq, velocity], axis=1)

X_data = np.array([add_velocity_features(seq) for seq in X_data])
print(f"После добавления скорости: {X_data.shape}")


le = LabelEncoder()
y_encoded = le.fit_transform(y_data)


train_mask = np.array([vid[0] == 'train' for vid in video_ids])
X_train_full = X_data[train_mask]
y_train_full = y_encoded[train_mask]
X_test = X_data[~train_mask]
y_test = y_encoded[~train_mask]

print(f"Train samples: {len(X_train_full)}, Test samples: {len(X_test)}")
print(f"Уникальных классов в train: {np.unique(y_train_full).shape[0]}")
print(f"Уникальных классов в test: {np.unique(y_test).shape[0]}")


train_classes = set(np.unique(y_train_full))
test_classes = set(np.unique(y_test))
missing_classes = train_classes - test_classes
if missing_classes:
    print(f"Предупреждение: В test отсутствуют классы: {[le.inverse_transform([c])[0] for c in missing_classes]}")


print("\n=== 5-FOLD CROSS-VALIDATION ===")
skf = StratifiedKFold(n_splits=min(Config.NUM_FOLDS, len(np.unique(y_train_full))),
                      shuffle=True, random_state=Config.RANDOM_SEED)
fold_histories = []
best_val_acc = 0
best_model = None
fold_val_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{min(Config.NUM_FOLDS, len(np.unique(y_train_full)))}")

    X_tr_fold = X_train_full[train_idx]
    y_tr_fold = y_train_full[train_idx]
    X_val_fold = X_train_full[val_idx]
    y_val_fold = y_train_full[val_idx]


    X_train_aug = []
    y_train_aug = []

    for seq, label in tqdm(zip(X_tr_fold, y_tr_fold), total=len(X_tr_fold)):
        aug_seqs = advanced_augmentation(seq)
        X_train_aug.extend(aug_seqs)
        y_train_aug.extend([label] * len(aug_seqs))

    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)

    print(f"После аугментации: {len(X_train_aug)} samples")


    y_train_cat = tf.keras.utils.to_categorical(y_train_aug, num_classes=len(top20))
    y_val_cat = tf.keras.utils.to_categorical(y_val_fold, num_classes=len(top20))


    tf.keras.backend.clear_session()
    model = create_model(len(top20))

    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train_aug, y_train_cat,
        validation_data=(X_val_fold, y_val_cat),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    fold_histories.append(history)

    val_pred = model.predict(X_val_fold)
    val_acc = balanced_accuracy_score(y_val_fold, np.argmax(val_pred, axis=1))
    val_std_acc = np.mean(np.argmax(val_pred, axis=1) == y_val_fold)
    fold_val_accs.append(val_acc)
    print(f"Fold {fold+1} - Accuracy: {val_std_acc:.2%}, Balanced Accuracy: {val_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        print(f"Новая лучшая модель! Balanced Accuracy: {best_val_acc:.2%}")

print(f"\nСредняя валидационная Balanced Accuracy: {np.mean(fold_val_accs):.2%} (+/- {np.std(fold_val_accs):.2%})")


print("\n=== ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ ===")
y_test_pred = best_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = y_test

test_acc = np.mean(y_test_pred_classes == y_test_true)
test_bal_acc = balanced_accuracy_score(y_test_true, y_test_pred_classes)

print(f"\nTest Accuracy: {test_acc:.2%}")
print(f"Test Balanced Accuracy: {test_bal_acc:.2%}")


unique_test_classes = np.unique(y_test_true)
test_class_names = [top20[i] for i in unique_test_classes]


y_test_true_mapped = y_test_true
y_test_pred_mapped = y_test_pred_classes

print("\nClassification Report (только классы, присутствующие в test):")
print(classification_report(y_test_true_mapped, y_test_pred_mapped,
                           labels=unique_test_classes,
                           target_names=test_class_names,
                           zero_division=0))


plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test_true, y_test_pred_classes, labels=range(len(top20)))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=top20, yticklabels=top20, cmap='Blues')
plt.title('Confusion Matrix - Test Set (All 20 Classes)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/confusion_matrix.png')
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for i, history in enumerate(fold_histories):
    plt.plot(history.history['accuracy'], label=f'Fold {i+1} Train', alpha=0.7)
plt.title('Training Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for i, history in enumerate(fold_histories):
    plt.plot(history.history['val_accuracy'], label=f'Fold {i+1} Validation', alpha=0.7)
plt.title('Validation Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/training_curves.png')
plt.show()

best_model.save('/content/drive/MyDrive/best_slovo_model.h5')
joblib.dump(le, '/content/drive/MyDrive/best_slovo_encoder.pkl')
joblib.dump(top20, '/content/drive/MyDrive/top20_classes.pkl')

print("\n" + "="*50)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print(f"Количество классов: {len(top20)}")
print(f"Размер последовательности: {Config.SEQUENCE_LEN}")
print(f"Количество признаков: {Config.FEATURES_WITH_VELOCITY}")
print(f"Train samples: {len(X_train_full)}")
print(f"Test samples: {len(X_test)}")
print(f"5-Fold Cross-Validation выполнен")
print(f"Лучшая Balanced Validation Accuracy: {best_val_acc:.2%}")
print(f"Финальная Test Balanced Accuracy: {test_bal_acc:.2%}")
print(f"Финальная Test Accuracy: {test_acc:.2%}")
print("="*50)


print("\n=== АНАЛИЗ ОШИБОК ===")
errors = np.where(y_test_pred_classes != y_test_true)[0]
print(f"Всего ошибок: {len(errors)}/{len(y_test)} ({len(errors)/len(y_test):.1%})")

if len(errors) > 0:
    error_pairs = [(y_test_true[i], y_test_pred_classes[i]) for i in errors]
    error_counts = Counter(error_pairs)
    print("\nТоп-10 частых ошибок:")
    for (true_label, pred_label), count in error_counts.most_common(10):
        print(f"  {top20[true_label]} -> {top20[pred_label]}: {count} раз")

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from google.colab import files
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os

model = tf.keras.models.load_model('/content/drive/MyDrive/best_slovo_model_new.keras')
le = joblib.load('/content/drive/MyDrive/best_slovo_encoder.pkl')
top20 = joblib.load('/content/drive/MyDrive/top20_classes.pkl')

if not os.path.exists('hand_landmarker.task'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'hand_landmarker.task'
    )

base = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

SEQUENCE_LEN = 40
TOTAL_FEATURES = 126

def normalize_landmarks(seq):
    normalized = []
    for frame in seq:
        points = frame.reshape(-1, 3)
        if np.all(points == 0):
            normalized.append(frame)
            continue
        wrist = points[0]
        centered = points - wrist
        scale = np.linalg.norm(centered, axis=1).max()
        if scale > 1e-6:
            centered = centered / scale
        normalized.append(centered.flatten())
    return np.array(normalized, dtype=np.float32)

def add_velocity(seq):
    velocity = np.diff(seq, axis=0)
    velocity = np.pad(velocity, ((0,1), (0,0)), 'constant')
    return np.concatenate([seq, velocity], axis=1)

def predict_gesture(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < SEQUENCE_LEN:
        cap.release()
        return f"Ошибка: видео слишком короткое ({total} кадров, нужно минимум {SEQUENCE_LEN})", 0

    indices = np.linspace(0, total-1, SEQUENCE_LEN, dtype=int)
    seq = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(img)

        vec = [0.0] * TOTAL_FEATURES
        if res.hand_landmarks:
            for hand_idx, hand in enumerate(res.hand_landmarks[:2]):
                base_idx = hand_idx * 63
                for i, lm in enumerate(hand[:21]):
                    vec[base_idx + i*3] = lm.x
                    vec[base_idx + i*3+1] = lm.y
                    vec[base_idx + i*3+2] = lm.z
        seq.append(vec)

    cap.release()

    while len(seq) < SEQUENCE_LEN:
        seq.append([0.0] * TOTAL_FEATURES)

    seq_array = np.array(seq, dtype=np.float32)
    seq_array = normalize_landmarks(seq_array)
    seq_array = add_velocity(seq_array)
    seq_array = seq_array.reshape(1, SEQUENCE_LEN, -1)

    pred = model.predict(seq_array, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred, axis=1)[0]

    return le.inverse_transform([pred_class])[0], confidence

print("Загрузите видео с жестом (рука должна быть чётко видна):")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\nОбработка: {filename}")
    result, conf = predict_gesture(filename)
    print(f"Распознанный жест: {result}")
    print(f"Уверенность: {conf:.2%}")