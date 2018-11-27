# Copyright (C) 2018 Project AGI
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

"""MNIST dataset using the tf.data module."""

import os
import gzip
import shutil
import tempfile
import logging

import numpy as np
import tensorflow as tf

from six.moves import urllib

from datasets.dataset import Dataset


class MNISTDataset(Dataset):
  """MNIST Dataset based on tf.data."""

  IMAGE_DIM = 28

  def __init__(self, directory):
    super(MNISTDataset, self).__init__(
        name='mnist',
        directory=directory,
        dataset_shape=[-1, 28, 28, 1],
        train_size=60000,
        test_size=10000,
        num_train_classes=10,
        num_test_classes=10,
        num_classes=10)

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for MNIST training data."""
    return self._dataset(
        self._directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', preprocess)

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for MNIST test data."""
    return self._dataset(
        self._directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', preprocess)

  def preprocess(self, image, preprocess):
    return image

  def _dataset(self, directory, images_file, labels_file, preprocess):
    """Download and parse MNIST dataset."""

    images_file = self._download(directory, images_file)
    labels_file = self._download(directory, labels_file)

    self._check_image_file_header(images_file)
    self._check_labels_file_header(labels_file)

    def decode_image(image):
      # Normalize from [0, 255] to [0.0, 1.0]
      image = tf.decode_raw(image, tf.uint8)
      image = tf.cast(image, tf.float32)
      image = tf.reshape(image, [self.IMAGE_DIM, self.IMAGE_DIM, 1])
      return image / 255.0

    def decode_label(label):
      label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
      label = tf.reshape(label, [])  # label is a scalar
      return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(images_file, np.prod([self.IMAGE_DIM, self.IMAGE_DIM, 1]),
                                              header_bytes=16)
    images = images.map(decode_image)
    images = images.map(lambda x: self.preprocess(x, preprocess))
    labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

  def _download(self, directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, filename)
    if tf.gfile.Exists(filepath):
      return filepath
    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + (
        filename + '.gz')
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    logging.info('Downloading %s to %s', url, zipped_filepath)
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
        tf.gfile.Open(filepath, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath

  def _check_image_file_header(self, filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
      magic = self._read32(f)
      self._read32(f)  # num_images, unused
      rows = self._read32(f)
      cols = self._read32(f)
      if magic != 2051:
        raise ValueError(
            'Invalid magic number %d in MNIST file %s' % (magic, f.name))
      if rows != self.IMAGE_DIM or cols != self.IMAGE_DIM:
        raise ValueError(
            'Invalid MNIST file %s: Expected %dx%d images, found %dx%d' %
            (f.name, self.IMAGE_DIM, self.IMAGE_DIM, rows, cols))

  def _check_labels_file_header(self, filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
      magic = self._read32(f)
      self._read32(f)  # num_items, unused
      if magic != 2049:
        raise ValueError(
            'Invalid magic number %d in MNIST file %s' % (magic, f.name))

  def _read32(self, bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]
