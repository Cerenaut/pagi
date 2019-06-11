# Copyright (C) 2019 Project AGI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""affNIST dataset using the tf.data module."""

import os
import glob
import zipfile
import tempfile
import logging

import numpy as np
import scipy.io as spio
import tensorflow as tf

from six.moves import urllib

from pagi.datasets.dataset import Dataset


class AffNISTDataset(Dataset):
  """affNIST Dataset based on tf.data."""

  IMAGE_SIZE_PX = 40

  def __init__(self, directory):
    super(AffNISTDataset, self).__init__(
        name='affnist',
        directory=directory,
        dataset_shape=[-1, 40, 40, 1],
        train_size=1920000,  # 60,000 * 32 transformations
        test_size=320000,  # 10,000 * 32 transformations
        num_train_classes=10,
        num_test_classes=10,
        num_classes=10)

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for affNIST training data."""
    return self._dataset('train', self._directory, 'sharded_train_0shifted_affnist.tfrecords')

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for affNIST test data."""
    return self._dataset('test', self._directory, 'sharded_test_0shifted_affnist.tfrecords')

  def _dataset(self, split, directory, data_file):
    """Download and parse affNIST dataset."""

    filenames = self._generate_filenames(split, directory, data_file)

    def parse_function(record):
      image_dim = self.IMAGE_SIZE_PX
      features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

      # Convert from a scalar string tensor (whose single string has
      # length image_pixel*image_pixel) to a uint8 tensor with shape
      # [image_pixel, image_pixel, 1].
      image = tf.decode_raw(features['image_raw'], tf.uint8)
      image = tf.reshape(image, [image_dim, image_dim, 1])
      image.set_shape([image_dim, image_dim, 1])

      # Convert from [0, 255] -> [-0.5, 0.5] floats.
      image = tf.cast(image, tf.float32) * (1. / 255)

      label = tf.cast(features['label'], tf.int32)

      return image, label

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function, num_parallel_calls=4)

    return dataset

  def _generate_filenames(self, split, directory, data_file):
    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, data_file)
    if not tf.gfile.Exists(dirpath):
      raise ValueError('Directory not found.')

    if data_file.startswith('sharded_'):
      sharded_file_format = data_file + '-*'
      tfrecords_list = glob.glob(os.path.join(dirpath, sharded_file_format))
      return tfrecords_list
    else:
      return filepath
