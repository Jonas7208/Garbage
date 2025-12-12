import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

# Modell laden
print("Lade Modell...")
model = load_model("models/best_model.keras")

print("\nOriginal-Modell:")
print(f"  - Anzahl Layer: {len(model.layers)}")
print(f"  - Trainierbare Parameter: {model.count_params()}")

# Alle BatchNormalization-Layer finden und einfrieren
bn_count = 0
for layer in model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
        bn_count += 1

print(f"\n✓ {bn_count} BatchNormalization-Layer eingefroren")

# Modell neu kompilieren (wichtig nach Änderung der trainable-Eigenschaft)
# Verwende einen neuen Optimizer, da der alte möglicherweise nicht serialisierbar ist
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # Passe an deine Loss-Funktion an
    metrics=['accuracy']
)

print("\nGefrorenes Modell:")
print(f"  - Anzahl Layer: {len(model.layers)}")
print(f"  - Trainierbare Parameter: {model.count_params()}")

# Validierung: Überprüfen ob alle BatchNorm-Layer eingefroren sind
bn_layers = [layer for layer in model.layers if isinstance(layer, BatchNormalization)]
all_frozen = all(not layer.trainable for layer in bn_layers)
print(f"\nValidierung:")
print(f"  - BatchNorm-Layer gefunden: {len(bn_layers)}")
print(f"  - Alle eingefroren: {'✓ Ja' if all_frozen else '✗ Nein'}")

# Speichern
print("\nSpeichere Modell...")
try:
    # Versuche mit Kompilierung zu speichern
    model.save("models/model_frozen.keras")
except NotImplementedError:
    # Falls Optimizer-Problem: Speichere nur Gewichte und Architektur
    print("⚠ Optimizer nicht serialisierbar - speichere nur Gewichte...")
    model.save("models/model_frozen.keras", save_format='keras', include_optimizer=False)

print("✓ BatchNormalization erfolgreich eingefroren und gespeichert.")

# Optional: Details zu den Layer-Typen anzeigen
print("\nLayer-Übersicht:")
layer_types = {}
for layer in model.layers:
    layer_type = type(layer).__name__
    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

for layer_type, count in sorted(layer_types.items()):
    print(f"  - {layer_type}: {count}")