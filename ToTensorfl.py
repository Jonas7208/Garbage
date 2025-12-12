import tensorflow as tf
import os
import numpy as np

print("=" * 60)
print("TensorFlow Lite Model Converter (Fixed)")
print("=" * 60)

# Modell laden
print("\n1. Lade Keras-Modell...")
model = tf.keras.models.load_model("models/best_model.keras")
print(f"✓ Modell geladen: {len(model.layers)} Layer")

# LÖSUNG: Erstelle ein Concrete Function für die Konvertierung
print("\n2. Erstelle Concrete Function...")

# Input Signature definieren
input_shape = model.input_shape
batch_size = 1  # Für Inference verwenden wir Batch Size 1


@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size] + list(input_shape[1:]), dtype=tf.float32)])
def model_fn(x):
    # BatchNorm im Inference-Modus erzwingen
    return model(x, training=False)


# Concrete Function erstellen
concrete_func = model_fn.get_concrete_function()

print(f"✓ Concrete Function erstellt: Input {concrete_func.inputs[0].shape}")

# Converter von Concrete Function erstellen
print("\n3. Erstelle TFLite Converter (von Concrete Function)...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Optimierungen
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Wichtig: Erlaube flexible Input Shapes
converter.experimental_new_converter = True

print("✓ Converter konfiguriert")

# Konvertierung durchführen
print("\n4. Konvertiere Modell zu TFLite...")
try:
    tflite_model = converter.convert()
    print("✓ Konvertierung erfolgreich!")
    uses_select_ops = False

except Exception as e:
    print(f"⚠ Fehler: {str(e)[:150]}")
    print("\nVersuche mit SELECT_TF_OPS...")

    # Neuversuch mit SELECT_TF_OPS
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_new_converter = True

    try:
        tflite_model = converter.convert()
        print("✓ Konvertierung erfolgreich (mit SELECT_TF_OPS)")
        uses_select_ops = True
    except Exception as e2:
        print(f"✗ Konvertierung fehlgeschlagen: {e2}")

        # Letzte Alternative: Ohne Optimierungen
        print("\nLetzte Alternative: Ohne Optimierungen...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        tflite_model = converter.convert()
        print("✓ Konvertierung erfolgreich (ohne Optimierungen)")
        uses_select_ops = True

# Modell speichern
print("\n5. Speichere TFLite-Modell...")
output_path = "models/model.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"✓ Gespeichert: {output_path}")

# Test: Modell laden und testen
print("\n6. Teste TFLite-Modell...")
interpreter = tf.lite.Interpreter(model_path=output_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ Modell erfolgreich geladen")
print(f"  Input:  {input_details[0]['shape']} ({input_details[0]['dtype']})")
print(f"  Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")

# Quick Test mit Random Input
test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
test_output = interpreter.get_tensor(output_details[0]['index'])
print(f"✓ Inference-Test erfolgreich: Output Shape {test_output.shape}")

# Statistiken
keras_size = os.path.getsize("models/best_model.keras") / (1024 * 1024)
tflite_size = os.path.getsize(output_path) / (1024 * 1024)
compression = (1 - tflite_size / keras_size) * 100

print("\n" + "=" * 60)
print("✓ Konvertierung abgeschlossen!")
print("=" * 60)
print(f"\nDateigrößen:")
print(f"  - Keras-Modell:  {keras_size:.2f} MB")
print(f"  - TFLite-Modell: {tflite_size:.2f} MB")
print(f"  - Kompression:   {compression:.1f}%")

if uses_select_ops:
    print("\n⚠ WICHTIG für Raspberry Pi:")
    print("  Modell verwendet SELECT_TF_OPS")
    print("  Installation: sudo pip3 install tensorflow")
else:
    print("\n✓ Modell verwendet nur Standard TFLite Ops")
    print("  Installation: sudo pip3 install tflite-runtime")

print("\n" + "=" * 60)