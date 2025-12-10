import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import json
import sys
from tqdm import tqdm

class Config:
    """Zentrale Konfiguration"""
    IMG_SIZE = (299, 299)
    MODEL_PATH = 'models/best_model.keras'  # oder 'models/final_resumed_model.keras'

    # Ordnerstruktur
    INPUT_DIR = '/Users/jonasgasparini/Desktop/UnsortierteBilder'  # Hier neue Bilder hochladen
    OUTPUT_HIGH_CONFIDENCE = 'sorted/high_confidence'  # >= 99% Vertrauen
    OUTPUT_LOW_CONFIDENCE = 'sorted/low_confidence'  # < 99% Vertrauen

    # Klassifizierungs-Einstellungen
    CONFIDENCE_THRESHOLD = 0.99  # 99% Schwellenwert

    # Unterstützte Bildformate
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']


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
        print("⚠️ Keine GPU gefunden - Klassifizierung läuft auf CPU")
    return False


def setup_directories(config):
    """Erstelle notwendige Verzeichnisse"""
    Path(config.INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_HIGH_CONFIDENCE).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_LOW_CONFIDENCE).mkdir(parents=True, exist_ok=True)

    print("✅ Verzeichnisse eingerichtet:")
    print(f"   📥 Input: {config.INPUT_DIR}")
    print(f"   ✅ High Confidence (≥99%): {config.OUTPUT_HIGH_CONFIDENCE}")
    print(f"   ⚠️  Low Confidence (<99%): {config.OUTPUT_LOW_CONFIDENCE}")


def load_model(model_path):
    """Lade trainiertes Modell"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modell geladen: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Fehler beim Laden des Modells: {e}")


def get_class_names(config):
    """Hole Klassennamen aus dem Trainingsverzeichnis"""
    train_dir = Path('dataset/train')
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"✅ Gefundene Klassen: {class_names}")
        return class_names
    else:
        # Fallback: Standard Müllklassen
        class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        print(f"⚠️  Verwende Standard-Klassen: {class_names}")
        return class_names


def get_image_files(input_dir, supported_formats):
    """Finde alle Bilddateien im Input-Verzeichnis"""
    image_files = []
    input_path = Path(input_dir)

    for fmt in supported_formats:
        image_files.extend(input_path.glob(f"*{fmt}"))
        image_files.extend(input_path.glob(f"*{fmt.upper()}"))

    return sorted(image_files)


def preprocess_image(img_path, img_size):
    """Lade und preprocesse ein einzelnes Bild"""
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisierung
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_and_sort_images(model, image_files, class_names, config):
    """Klassifiziere Bilder und sortiere sie nach Confidence"""
    results = {
        'high_confidence': [],
        'low_confidence': [],
        'errors': []
    }

    print(f"\n🔍 Klassifiziere {len(image_files)} Bilder...\n")

    for img_path in tqdm(image_files, desc="Fortschritt"):
        try:
            # Bild preprocessen
            img_array = preprocess_image(img_path, config.IMG_SIZE)

            # Vorhersage
            predictions = model.predict(img_array, verbose=0)
            confidence = float(np.max(predictions))
            predicted_class_idx = int(np.argmax(predictions))
            predicted_class = class_names[predicted_class_idx]

            # Bestimme Zielordner
            if confidence >= config.CONFIDENCE_THRESHOLD:
                base_output_dir = Path(config.OUTPUT_HIGH_CONFIDENCE)
                category = 'high_confidence'
            else:
                base_output_dir = Path(config.OUTPUT_LOW_CONFIDENCE)
                category = 'low_confidence'

            # Erstelle Klassenunterordner
            class_output_dir = base_output_dir / predicted_class
            class_output_dir.mkdir(parents=True, exist_ok=True)

            # Kopiere Bild mit neuem Namen (füge Confidence hinzu)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_filename = f"{predicted_class}_{confidence:.4f}_{timestamp}{img_path.suffix}"
            destination = class_output_dir / new_filename

            shutil.copy2(img_path, destination)

            # Speichere Ergebnis
            result_entry = {
                'original_path': str(img_path),
                'new_path': str(destination),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': {
                    class_names[i]: float(predictions[0][i])
                    for i in range(len(class_names))
                }
            }

            results[category].append(result_entry)

        except Exception as e:
            error_entry = {
                'file': str(img_path),
                'error': str(e)
            }
            results['errors'].append(error_entry)
            print(f"\n⚠️  Fehler bei {img_path.name}: {e}")

    return results


def print_summary(results):
    """Zeige Zusammenfassung der Klassifizierung"""
    print("\n" + "=" * 70)
    print("📊 KLASSIFIZIERUNGS-ZUSAMMENFASSUNG")
    print("=" * 70)

    high_conf = len(results['high_confidence'])
    low_conf = len(results['low_confidence'])
    errors = len(results['errors'])
    total = high_conf + low_conf + errors

    print(f"\n✅ High Confidence (≥99%): {high_conf} Bilder ({high_conf / total * 100:.1f}%)")
    print(f"⚠️  Low Confidence (<99%):  {low_conf} Bilder ({low_conf / total * 100:.1f}%)")

    if errors > 0:
        print(f"❌ Fehler:                  {errors} Bilder ({errors / total * 100:.1f}%)")

    print(f"\n📁 Gesamt verarbeitet:      {total} Bilder")

    # Detaillierte Klassenverteilung für High Confidence
    if high_conf > 0:
        print("\n" + "-" * 70)
        print("High Confidence Verteilung:")
        print("-" * 70)
        class_counts = {}
        for item in results['high_confidence']:
            cls = item['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        for cls, count in sorted(class_counts.items()):
            print(f"   {cls:.<20} {count:>3} Bilder")

    # Detaillierte Klassenverteilung für Low Confidence
    if low_conf > 0:
        print("\n" + "-" * 70)
        print("Low Confidence Verteilung:")
        print("-" * 70)
        class_counts = {}
        avg_confidence = {}
        for item in results['low_confidence']:
            cls = item['predicted_class']
            conf = item['confidence']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in avg_confidence:
                avg_confidence[cls] = []
            avg_confidence[cls].append(conf)

        for cls, count in sorted(class_counts.items()):
            avg_conf = np.mean(avg_confidence[cls]) * 100
            print(f"   {cls:.<20} {count:>3} Bilder (Ø {avg_conf:.1f}% Confidence)")


def save_results_json(results, output_file='classification_results.json'):
    """Speichere detaillierte Ergebnisse als JSON"""
    timestamp = datetime.now().isoformat()

    output_data = {
        'timestamp': timestamp,
        'summary': {
            'total_images': len(results['high_confidence']) + len(results['low_confidence']) + len(results['errors']),
            'high_confidence_count': len(results['high_confidence']),
            'low_confidence_count': len(results['low_confidence']),
            'error_count': len(results['errors'])
        },
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detaillierte Ergebnisse gespeichert: {output_file}")


def main():
    """Hauptfunktion"""
    print("=" * 70)
    print("🗑️  Automatische Müll-Klassifizierung und Sortierung")
    print("=" * 70)

    config = Config()

    try:
        # Setup
        setup_gpu()
        setup_directories(config)

        # Prüfe ob Input-Verzeichnis Bilder enthält
        image_files = get_image_files(config.INPUT_DIR, config.SUPPORTED_FORMATS)

        if not image_files:
            print(f"\n⚠️  Keine Bilder im Ordner '{config.INPUT_DIR}' gefunden!")
            print(f"   Unterstützte Formate: {', '.join(config.SUPPORTED_FORMATS)}")
            print(f"\n💡 Bitte Bilder in den '{config.INPUT_DIR}' Ordner hochladen.")
            return

        print(f"\n📸 {len(image_files)} Bilder gefunden")

        # Lade Modell
        print("\n" + "=" * 70)
        model = load_model(config.MODEL_PATH)

        # Hole Klassennamen
        class_names = get_class_names(config)

        # Klassifiziere und sortiere
        print("\n" + "=" * 70)
        results = classify_and_sort_images(model, image_files, class_names, config)

        # Zeige Zusammenfassung
        print_summary(results)

        # Speichere detaillierte Ergebnisse
        save_results_json(results)

        print("\n" + "=" * 70)
        print("✅ Klassifizierung abgeschlossen!")
        print("=" * 70)
        print(f"\n📁 Sortierte Bilder:")
        print(f"   ✅ High Confidence: {config.OUTPUT_HIGH_CONFIDENCE}")
        print(f"   ⚠️  Low Confidence:  {config.OUTPUT_LOW_CONFIDENCE}")

        # Optional: Originale löschen
        delete_originals = input("\n🗑️  Originale aus 'uploads' löschen? (j/n) [n]: ").lower()
        if delete_originals in ['j', 'ja', 'y', 'yes']:
            for img_file in image_files:
                try:
                    img_file.unlink()
                except Exception as e:
                    print(f"⚠️  Konnte {img_file.name} nicht löschen: {e}")
            print("✅ Originale gelöscht")

        print("=" * 70)

    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()