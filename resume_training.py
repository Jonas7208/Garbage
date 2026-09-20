from training_metrics import classification_metrics, save_validation_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import sys


class Config:
    """Zentrale Konfiguration"""
    IMG_SIZE = (299, 299)
    BATCH_SIZE = 32
    ADDITIONAL_EPOCHS = 10
    LEARNING_RATE = 0.00005
    TRAIN_DIR = 'dataset/train'
    NUM_CLASSES = 6
    MODEL_PATH = 'models/best_model.keras'

    # Neue Modellpfade
    RESUMED_BEST = 'models/resumed_best_model.keras'
    FINAL_RESUMED = 'models/final_resumed_model.keras'

    # Fine-Tuning Einstellungen
    UNFREEZE_FROM_LAYER = 200
    AUTO_UNFREEZE = False  # Automatisch entfrieren ohne Nachfrage


def setup_gpu():
    """Konfiguriere GPU für optimale Performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU gefunden: {len(gpus)} Gerät(e)")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU-Konfiguration fehlgeschlagen: {e}")
    else:
        print("⚠️ Keine GPU gefunden - Training läuft auf CPU")
    return False


def validate_directories(config):
    """Überprüfe ob alle benötigten Verzeichnisse existieren"""
    train_path = Path(config.TRAIN_DIR)
    model_path = Path(config.MODEL_PATH)

    if not train_path.exists():
        raise FileNotFoundError(f"❌ Trainingsverzeichnis nicht gefunden: {train_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Modell nicht gefunden: {model_path}")

    # Erstelle Ausgabeverzeichnisse falls nicht vorhanden
    Path('models').mkdir(exist_ok=True)
    Path('logs/fit').mkdir(parents=True, exist_ok=True)

    print("✅ Alle Verzeichnisse validiert")


def load_model(model_path):
    """Lade Modell mit Fehlerbehandlung"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modell geladen von: {model_path}")

        # Zeige Modell-Informationen
        print("\n📊 Modell-Info:")
        print(f"   Total Parameters: {model.count_params():,}")

        # Korrekte Methode für trainierbare Parameter
        trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"   Trainable Parameters: {trainable_count:,}")

        return model
    except Exception as e:
        raise RuntimeError(f"❌ Fehler beim Laden des Modells: {e}")


def create_data_generators(config):
    """Erstelle Daten-Generatoren mit Fehlerbehandlung"""
    print("\n📊 Bereite Daten vor...")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            config.TRAIN_DIR,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = val_datagen.flow_from_directory(
            config.TRAIN_DIR,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        print(f"✅ Trainingssamples: {train_generator.samples}")
        print(f"✅ Validierungssamples: {validation_generator.samples}")
        print(f"✅ Klassen: {list(train_generator.class_indices.keys())}")

        return train_generator, validation_generator

    except Exception as e:
        raise RuntimeError(f"❌ Fehler beim Erstellen der Generatoren: {e}")


def calculate_class_weights(train_generator, config):
    """Berechne Class Weights für unbalancierte Daten"""
    print("\n⚖️ Berechne Class Weights...")

    class_counts = np.zeros(config.NUM_CLASSES)
    for class_name, class_idx in train_generator.class_indices.items():
        class_dir = Path(config.TRAIN_DIR) / class_name
        class_counts[class_idx] = len(list(class_dir.iterdir()))

    total_samples = class_counts.sum()
    class_weights = {
        i: total_samples / (config.NUM_CLASSES * count)
        for i, count in enumerate(class_counts)
    }

    # Zeige Class Distribution
    print("\n📊 Klassenverteilung:")
    for class_name, class_idx in train_generator.class_indices.items():
        print(f"   {class_name}: {int(class_counts[class_idx])} Bilder (Weight: {class_weights[class_idx]:.2f})")

    return class_weights


def unfreeze_layers(model, config):
    """Entfiere mehr Layers für besseres Fine-Tuning"""
    print("\n🔓 Fine-Tuning Konfiguration...")

    # Automatische oder manuelle Entscheidung
    if not config.AUTO_UNFREEZE:
        user_input = input("Möchtest du mehr Layers trainieren? (j/n) [Standard: n]: ").lower()
        if user_input not in ['j', 'ja', 'y', 'yes']:
            print("⏭️ Überspringe Layer-Entfrierung")
            return

    # Finde Base Model (InceptionV3 oder ähnlich)
    base_model_found = False
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > config.UNFREEZE_FROM_LAYER:
            base_model_found = True
            print(f"✅ Base Model gefunden: {layer.name} mit {len(layer.layers)} Layers")

            # Entfiere Layers ab bestimmtem Index
            for sublayer in layer.layers[config.UNFREEZE_FROM_LAYER:]:
                sublayer.trainable = True

            trainable_layers = len([l for l in layer.layers if l.trainable])
            print(f"✅ {trainable_layers} Layers des Base Models sind jetzt trainierbar")
            break

    if not base_model_found:
        print("⚠️ Kein Base Model mit genug Layers gefunden")

    # Zähle trainierbare Parameter
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"   Neue trainierbare Parameter: {trainable_count:,}")


def setup_callbacks(config):
    """Konfiguriere Training Callbacks"""
    print("\n⚙️ Konfiguriere Callbacks...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{timestamp}_resume"

    callbacks = [
        ModelCheckpoint(
            config.RESUMED_BEST,
            monitor='val_macro_f1',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_macro_f1',
            mode='max',
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
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    return callbacks, log_dir


def plot_training_history(history, save_path='training_history_resumed.png'):
    """Visualisiere Trainingsverlauf"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Top-2 Accuracy
    if 'top_2_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_2_accuracy'], label='Training')
        axes[1, 0].plot(history.history['val_top_2_accuracy'], label='Validation')
        axes[1, 0].set_title('Top-2 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-2 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Hauptmetrik für die Modellauswahl
    if 'macro_f1' in history.history:
        axes[1, 1].plot(history.history['macro_f1'], label='Training')
        axes[1, 1].plot(history.history['val_macro_f1'], label='Validation')
        axes[1, 1].set_title('Macro-F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Trainingshistorie gespeichert: {save_path}")
    plt.close()


def save_training_summary(results_before, results_after, config, log_dir):
    """Speichere Trainings-Zusammenfassung als JSON"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'additional_epochs': config.ADDITIONAL_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'unfreeze_from_layer': config.UNFREEZE_FROM_LAYER,
        },
        'results': {
            'before': {
                'accuracy': float(results_before[1]),
                'macro_f1': float(results_before[3]),
                'loss': float(results_before[0]),
            },
            'after': {
                'accuracy': float(results_after[1]),
                'macro_f1': float(results_after[3]),
                'loss': float(results_after[0]),
            },
            'improvement': {
                'accuracy': float(results_after[1] - results_before[1]),
                'accuracy_percent': float((results_after[1] - results_before[1]) * 100),
                'macro_f1': float(results_after[3] - results_before[3]),
            }
        },
        'log_dir': log_dir
    }

    summary_path = 'training_summary_resumed.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Trainings-Zusammenfassung gespeichert: {summary_path}")


def main():
    """Hauptfunktion für Resume Training"""
    print("=" * 60)
    print("🔄 Garbage Classification Model - Weitertrainieren")
    print("=" * 60)

    config = Config()

    try:
        # Setup
        setup_gpu()
        validate_directories(config)

        # Lade Modell
        print("\n" + "=" * 60)
        print("📦 SCHRITT 1: Modell laden")
        print("=" * 60)
        model = load_model(config.MODEL_PATH)

        # Daten vorbereiten
        print("\n" + "=" * 60)
        print("📊 SCHRITT 2: Daten vorbereiten")
        print("=" * 60)
        train_gen, val_gen = create_data_generators(config)
        class_weights = calculate_class_weights(train_gen, config)

        # Optional: Layers entfrieren
        print("\n" + "=" * 60)
        print("🔓 SCHRITT 3: Fine-Tuning konfigurieren")
        print("=" * 60)
        unfreeze_layers(model, config)

        # Modell neu kompilieren
        print("\n" + "=" * 60)
        print("⚙️ SCHRITT 4: Modell kompilieren")
        print("=" * 60)
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=classification_metrics()
        )
        print(f"✅ Learning Rate: {config.LEARNING_RATE}")

        # Callbacks einrichten
        callbacks, log_dir = setup_callbacks(config)

        # Evaluiere vor Training
        print("\n" + "=" * 60)
        print("📊 SCHRITT 5: Evaluierung vor Training")
        print("=" * 60)
        results_before = model.evaluate(val_gen, verbose=0)
        print(f"   Val-Accuracy: {results_before[1] * 100:.2f}%")
        print(f"   Val-Loss: {results_before[0]:.4f}")

        # Training
        print("\n" + "=" * 60)
        print(f"🏋️ SCHRITT 6: Weitertraining ({config.ADDITIONAL_EPOCHS} Epochen)")
        print("=" * 60)

        history = model.fit(
            train_gen,
            epochs=config.ADDITIONAL_EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Evaluiere nach Training
        print("\n" + "=" * 60)
        print("📊 SCHRITT 7: Finale Evaluierung")
        print("=" * 60)

        best_model = tf.keras.models.load_model(config.RESUMED_BEST)
        results_after = best_model.evaluate(val_gen, verbose=1)

        # Ergebnisse anzeigen
        print("\n" + "=" * 60)
        print("✅ FINALE METRIKEN")
        print("=" * 60)
        print(f"   Vorher:      {results_before[1] * 100:.2f}%")
        print(f"   Nachher:     {results_after[1] * 100:.2f}%")
        print(f"   Macro-F1 vorher: {results_before[3]:.4f}")
        print(f"   Macro-F1 nachher: {results_after[3]:.4f}")
        improvement = (results_after[3] - results_before[3]) * 100
        print(f"   Macro-F1 Änderung (Prozentpunkte): {improvement:+.2f}%")

        if improvement > 0:
            print("\n🎉 Modell hat sich verbessert!")
        elif improvement < 0:
            print("\n⚠️ Modell hat sich verschlechtert - überprüfe Hyperparameter")
        else:
            print("\n📊 Keine signifikante Änderung")

        # Speichere finales Modell
        best_model.save(config.FINAL_RESUMED)
        print("\n💾 Modelle gespeichert:")
        print(f"   ✅ {config.RESUMED_BEST} (bestes während Training)")
        print(f"   ✅ {config.FINAL_RESUMED} (finales Modell)")

        # Visualisierung und Zusammenfassung
        plot_training_history(history)
        save_training_summary(results_before, results_after, config, log_dir)
        class_names = [name for name, index in sorted(val_gen.class_indices.items(), key=lambda item: item[1])]
        save_validation_report(best_model, val_gen, class_names, Path(log_dir) / 'validation_report.json')

        print("\n" + "=" * 60)
        print("🎉 Weitertraining erfolgreich abgeschlossen!")
        print("=" * 60)
        print(f"\n📊 TensorBoard starten mit:")
        print(f"   tensorboard --logdir={log_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()