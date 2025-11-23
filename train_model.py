import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from datetime import datetime
from pathlib import Path
import os

print("=" * 60)
print("🚀 Garbage Classification Model - Training")
print("=" * 60)

# ============================================================
# KONFIGURATION
# ============================================================
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.0001
TRAIN_DIR = 'dataset/train'
NUM_CLASSES = 6

# ============================================================
# GPU Setup
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU gefunden: {len(gpus)} Gerät(e)")
    except RuntimeError as e:
        print(f"⚠️ GPU Setup Fehler: {e}")
else:
    print("⚠️ Keine GPU gefunden - Training auf CPU")

# ============================================================
# Dataset Verzeichnis prüfen
# ============================================================
if not os.path.exists(TRAIN_DIR):
    print(f"\n❌ FEHLER: '{TRAIN_DIR}' nicht gefunden!")
    print("\n💡 Alternative Pfade:")
    alternatives = ['dataset-resized/train', 'dataset', 'data/train']
    for path in alternatives:
        if os.path.exists(path):
            print(f"   ✅ {path} gefunden!")
            TRAIN_DIR = path
            break
        else:
            print(f"   ❌ {path}")

    if not os.path.exists(TRAIN_DIR):
        print("\n💡 Passe TRAIN_DIR im Code an!")
        exit(1)

print(f"✅ Dataset: {TRAIN_DIR}")
os.makedirs('models', exist_ok=True)
os.makedirs('logs/fit', exist_ok=True)

# ============================================================
# Daten-Generatoren
# ============================================================
print("\n📊 Erstelle Daten-Generatoren...")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"✅ Training: {train_generator.samples} Bilder")
print(f"✅ Validation: {validation_generator.samples} Bilder")
print(f"✅ Klassen: {list(train_generator.class_indices.keys())}")

# Passe NUM_CLASSES automatisch an
NUM_CLASSES = len(train_generator.class_indices)

# ============================================================
# Class Weights
# ============================================================
print("\n⚖️ Berechne Class Weights...")

class_counts = np.zeros(NUM_CLASSES)
for class_name, class_idx in train_generator.class_indices.items():
    class_dir = os.path.join(TRAIN_DIR, class_name)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_idx] = len(files)

total = class_counts.sum()
class_weights = {i: total / (NUM_CLASSES * count) if count > 0 else 1.0
                 for i, count in enumerate(class_counts)}

print("Class Weights:")
for name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
    print(f"   {name}: {int(class_counts[idx])} Bilder (Weight: {class_weights[idx]:.2f})")

# ============================================================
# Modell erstellen
# ============================================================
print("\n📦 Erstelle Modell...")

# Base Model
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

# Custom Top Layers
# L2 reduziert da BatchNorm verwendet wird (siehe Paper: Faktor 5 reduzieren)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.002))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.002))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(f"✅ Modell erstellt: {model.count_params():,} Parameter")

# ============================================================
# Callbacks
# ============================================================
print("\n⚙️ Konfiguriere Callbacks...")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/fit/{timestamp}"

callbacks = [
    ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# ============================================================
# PHASE 1: Initial Training
# ============================================================
print("\n" + "=" * 60)
print("🏋️ PHASE 1: Training mit gefrorenem Base Model")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print(f"Epochen: {INITIAL_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")

history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# PHASE 2: Fine-Tuning
# ============================================================
print("\n" + "=" * 60)
print("🔥 PHASE 2: Fine-Tuning")
print("=" * 60)

# Entfrier obere Layers
base_model.trainable = True
for layer in base_model.layers[:249]:
    layer.trainable = False

trainable = len([l for l in base_model.layers if l.trainable])
print(f"Trainierbare Layers: {trainable}/{len(base_model.layers)}")

# Neu kompilieren mit niedriger LR
fine_tune_lr = LEARNING_RATE / 10
model.compile(
    optimizer=Adam(learning_rate=fine_tune_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print(f"Learning Rate: {fine_tune_lr}")

initial_epoch = len(history1.history['loss'])
total_epochs = initial_epoch + FINE_TUNE_EPOCHS

history2 = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=initial_epoch,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# Finale Evaluierung
# ============================================================
print("\n" + "=" * 60)
print("📊 FINALE EVALUIERUNG")
print("=" * 60)

best_model = tf.keras.models.load_model('models/best_model.keras')
results = best_model.evaluate(validation_generator, verbose=1)

print("\n✅ FINALE METRIKEN:")
print(f"   Loss:           {results[0]:.4f}")
print(f"   Accuracy:       {results[1]:.4f} ({results[1] * 100:.2f}%)")
print(f"   Top-2 Accuracy: {results[2]:.4f} ({results[2] * 100:.2f}%)")

# Speichere finales Modell
model.save('models/final_model.keras')
print("\n💾 Modelle gespeichert:")
print("   ✅ models/best_model.keras")
print("   ✅ models/final_model.keras")

print("\n🎉 Training abgeschlossen!")
print("=" * 60)
print(f"\n📊 TensorBoard: tensorboard --logdir={log_dir}")
print("=" * 60)