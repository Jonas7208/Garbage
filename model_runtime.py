"""Load classification models without compiling their training state."""
from pathlib import Path
import tempfile
import threading

import numpy as np
import tensorflow as tf


def has_rescaling(model):
    return any(isinstance(layer, tf.keras.layers.Rescaling) or
               (hasattr(layer, "layers") and has_rescaling(layer))
               for layer in model.layers)


class Classifier:
    def __init__(self, model=None, interpreter=None):
        self.model = model
        self.interpreter = interpreter
        self.lock = threading.Lock()
        self.embedded_rescaling = has_rescaling(model) if model is not None else None
        if interpreter is not None:
            interpreter.allocate_tensors()
            inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Das Modell muss genau einen Bildeingang und einen Klassenausgang haben.")
            self.input_info, self.output_info = inputs[0], outputs[0]
            input_shape, output_shape = inputs[0]["shape"], outputs[0]["shape"]
        else:
            input_shape, output_shape = model.input_shape, model.output_shape
            if isinstance(input_shape, list) or isinstance(output_shape, list):
                raise ValueError("Modelle mit mehreren Ein- oder Ausgängen werden nicht unterstützt.")
        if (len(input_shape) != 4 or input_shape[-1] != 3 or
                input_shape[0] not in (None, 1) or
                any(x is None or x <= 0 for x in input_shape[1:3])):
            raise ValueError("Erwartet wird ein RGB-Bildmodell mit fester Bildgröße und Batchgröße 1 oder variabel.")
        if len(output_shape) != 2 or output_shape[-1] != 6:
            raise ValueError("Das Modell muss genau sechs Müllklassen ausgeben.")
        self.image_size = (int(input_shape[2]), int(input_shape[1]))

    def prepare_image(self, image, scaling):
        pixels = np.asarray(image.convert("RGB").resize(self.image_size), dtype=np.float32)
        if scaling == "0–1":
            pixels /= 255.0
        elif scaling == "−1–1":
            pixels = pixels / 127.5 - 1.0
        return pixels[None, ...]

    def predict(self, pixels, verbose=0):
        with self.lock:
            if self.model is not None:
                result = np.asarray(self.model(pixels, training=False))
            else:
                info = self.input_info
                if np.issubdtype(info["dtype"], np.integer):
                    scale, zero = info["quantization"]
                    if scale <= 0:
                        raise ValueError("Ungültige Eingabequantisierung im TFLite-Modell.")
                    bounds = np.iinfo(info["dtype"])
                    pixels = np.clip(np.rint(pixels / scale + zero), bounds.min, bounds.max)
                self.interpreter.set_tensor(info["index"], pixels.astype(info["dtype"]))
                self.interpreter.invoke()
                result = self.interpreter.get_tensor(self.output_info["index"])
                if np.issubdtype(result.dtype, np.integer):
                    scale, zero = self.output_info["quantization"]
                    result = (result.astype(np.float32) - zero) * scale
        if not np.all(np.isfinite(result)) or np.any(result < 0) or not np.allclose(result.sum(axis=1), 1, atol=0.03):
            raise ValueError("Das Modell muss Klassenwahrscheinlichkeiten ausgeben (z. B. Softmax).")
        return result


def load_classifier(path):
    path = Path(path)
    if path.suffix.lower() == ".tflite":
        return Classifier(interpreter=tf.lite.Interpreter(model_path=str(path)))
    if path.suffix.lower() != ".keras":
        raise ValueError("Bitte eine .keras- oder .tflite-Datei auswählen.")
    return Classifier(model=tf.keras.models.load_model(path, compile=False, safe_mode=True))


def load_uploaded_classifier(content, suffix):
    with tempfile.TemporaryDirectory(prefix="garbage-model-") as directory:
        path = Path(directory) / ("uploaded" + suffix)
        path.write_bytes(content)
        return load_classifier(path)
