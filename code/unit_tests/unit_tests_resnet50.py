import mltest
import kaggle_scripts_resnet50
import tensorflow as tf
import numpy as np

def setup():
  mltest.setup()

def resnet50_mltest_suite():

  input_tensor = tf.placeholder(tf.float32, (None, 100))
  label_tensor = tf.placeholder(tf.int32, (None))

  model = kaggle_scripts_resnet50.build_model(input_tensor, label_tensor)

  feed_dict = {
      input_tensor: np.random.normal(size=(10, 100)),
      label_tensor: np.random.randint((100))
  }

  mltest.test_suite(
      model.prediction,
      model.train_op,
      feed_dict=feed_dict)

def test_range():
  model = kaggle_scripts_resnet50.build_model()
  mltest.test_suite(
    model.logits,
    model.train_op,
    output_range=(0,1))
