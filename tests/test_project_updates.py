import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tensorflow as tf
from PIL import Image
from training_metrics import classification_metrics, report_from_predictions, save_validation_report
from model_runtime import load_classifier, load_uploaded_classifier

class ChangesTest(unittest.TestCase):
 def test_metrics(self):
  labels=np.eye(6,dtype=np.float32)[[0,0,1,2,3,4,5]]
  predictions=np.eye(6,dtype=np.float32)[[0,1,1,2,3,4,4]]
  m=classification_metrics()[-1]; m.update_state(labels[:3],predictions[:3]);m.update_state(labels[3:],predictions[3:])
  report=report_from_predictions(labels.argmax(1),predictions.argmax(1),['a','b','c','d','e','f'])
  self.assertAlmostEqual(float(m.result()),report['macro_f1'],places=6)
  self.assertEqual(report['confusion_matrix'][0][1],1)
  self.assertEqual(report['per_class']['f']['recall'],0)
 def test_upload_and_report(self):
  x=tf.keras.Input((8,8,3)); y=tf.keras.layers.Rescaling(1/255)(x);y=tf.keras.layers.GlobalAveragePooling2D()(y);y=tf.keras.layers.Dense(6,activation='softmax')(y);model=tf.keras.Model(x,y)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'m.keras';model.save(path)
   runtime=load_uploaded_classifier(path.read_bytes(),'.keras')
   self.assertTrue(runtime.embedded_rescaling)
   pixels=runtime.prepare_image(Image.new('RGBA',(12,12),'white'),'0–255')
   np.testing.assert_allclose(runtime.predict(pixels),model(pixels),atol=1e-6)
   report=save_validation_report(model,[(pixels,np.eye(6,dtype=np.float32)[[1]])],list('abcdef'),Path(d)/'report.json')
   self.assertEqual(sum(v['support'] for v in report['per_class'].values()),1)
 def test_invalid_upload(self):
  with self.assertRaises(Exception):load_uploaded_classifier(b'not a model','.keras')
 def test_real_models(self):
  root=Path(__file__).resolve().parents[1]/'models'
  for name in ['best_model.keras','model.tflite']:
   if not (root/name).exists():
    continue
   runtime=load_classifier(root/name)
   pixels=runtime.prepare_image(Image.new('RGB',(299,299),'white'),'0–1')
   result=runtime.predict(pixels)
   self.assertEqual(result.shape,(1,6));self.assertTrue(np.all(np.isfinite(result)))
   print('REAL MODEL OK',name,'embedded rescaling',runtime.embedded_rescaling,flush=True)
 def test_scripts(self):
  for name in ['train_model.py','resume_training.py','garbage_app.py']:
   ast.parse(Path(name).read_text())
  for name in ['train_model.py','resume_training.py']:
   text=Path(name).read_text();self.assertIn("monitor='val_macro_f1'",text);self.assertIn('save_validation_report(',text)
if __name__=='__main__':unittest.main(verbosity=2)
