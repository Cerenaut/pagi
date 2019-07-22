from __future__ import absolute_import, division, print_function

import tensorflow as tf
from pagi.utils import data_utils

tf.enable_eager_execution()

dataset_shape = [1, 28, 28, 1]
filename = './train_new_dataset.tfrecords'

dataset = data_utils.read_subset(filename, dataset_shape)

for idx, (image, label) in enumerate(dataset):
  print(str(idx), str(label))

