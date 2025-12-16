import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import os

# Config
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LR = 0.0001
TRAIN_DIR = 'dataset/train'



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    # Apple Metal GPU aktivieren
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Metal GPU gefunden: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Läuft auf CPU (kein Metal GPU gefunden)")
except Exception as e:
    print(f"GPU-Setup übersprungen: {e}")

# Find dataset
if not os.path.exists(TRAIN_DIR):
    for path in ['dataset-resized/train', 'dataset', 'data/train']:
        if os.path.exists(path):
            TRAIN_DIR = path
            break

os.makedirs('models', exist_ok=True)

# Load datasets using tf.keras.utils
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=123,
    interpolation='bilinear'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123,
    interpolation='bilinear'
)

# Get class info BEFORE any transformations
class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"\nClasses: {class_names}")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

# Apply augmentation with error handling
def augment_and_validate(image, label):
    try:
        image = data_augmentation(image, training=True)
        # Ensure valid range
        image = tf.clip_by_value(image, 0.0, 255.0)
    except:
        pass  # Keep original if augmentation fails
    return image, label

train_dataset = train_dataset.map(augment_and_validate, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# Calculate class weights
class_counts = np.zeros(NUM_CLASSES)
for class_name in class_names:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_idx = class_names.index(class_name)
        class_counts[class_idx] = len(files)

total = class_counts.sum()
class_weights = {i: total / (NUM_CLASSES * count) if count > 0 else 1.0
                 for i, count in enumerate(class_counts)}

# Build model
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = Rescaling(1./255)(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.002))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.002))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)


callbacks = [
    ModelCheckpoint('models/best_model.keras', monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1)
]

print("\nPhase 1: Initial training")
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

history1 = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks,
    class_weight=class_weights
)

print("\nPhase 2: Fine-tuning")
base_model.trainable = True
for layer in base_model.layers[:249]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

history2 = model.fit(
    train_dataset,
    epochs=len(history1.history['loss']) + FINE_TUNE_EPOCHS,
    initial_epoch=len(history1.history['loss']),
    validation_data=validation_dataset,
    callbacks=callbacks,
    class_weight=class_weights
)

print("\nFinal evaluation")
best_model = tf.keras.models.load_model('models/best_model.keras')
results = best_model.evaluate(validation_dataset, verbose=0)

print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"Top-2 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")

model.save('models/final_model.keras')
print("\nTraining complete")