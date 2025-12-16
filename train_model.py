import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from datetime import datetime

IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.0001
TRAIN_DIR = 'dataset/train'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Metal GPU gefunden: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Läuft auf CPU")
except Exception as e:
    print(f"GPU-Setup übersprungen: {e}")

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=123
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"\nKlassen ({num_classes}): {class_names}")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

def augment_images(image, label):
    image = data_augmentation(image, training=True)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

train_dataset = train_dataset.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

print("\nBerechne Class Weights...")
class_counts = np.zeros(num_classes)

for class_name in class_names:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_idx = class_names.index(class_name)
        class_counts[class_idx] = len(files)
        print(f"  {class_name}: {len(files)} Bilder")

total = class_counts.sum()
class_weights = {i: total / (num_classes * count) if count > 0 else 1.0
                 for i, count in enumerate(class_counts)}

print("\nErstelle Modell mit InceptionV3...")

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
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

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

print("PHASE 1: Initial Training (Base Model eingefroren)")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
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


print("PHASE 2: Fine-Tuning (Top-Layer trainierbar)")

base_model.trainable = True
for layer in base_model.layers[:249]:
    layer.trainable = False

print(f"Trainierbare Layer: {sum([1 for layer in base_model.layers if layer.trainable])}")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
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



best_model = tf.keras.models.load_model('models/best_model.keras')
results = best_model.evaluate(validation_dataset, verbose=0)

print(f"\nErgebnisse:")
print(f"  Loss: {results[0]:.4f}")
print(f"  Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"  Top-2 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")

model.save('models/final_model.keras')

print(f"Beste Modell: models/best_model.keras")
print(f"TensorBoard Logs: {log_dir}")
print(f"\nStarte TensorBoard: tensorboard --logdir=logs")