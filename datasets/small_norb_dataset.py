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


class SmallNORBDataset(Dataset):
  """smallNORB Dataset based on tf.data."""

  SMALLNORB_SIZE = 96
  SMALLNORB_DEPTH = 2

  IMAGE_DIM = 48  # downsample to 48 or 32
  IMAGE_DEPTH = 1  # 1: first image in the pair, 2: use stereo pair

  def __init__(self, directory):
    super(SmallNORBDataset, self).__init__(
        name='small_norb',
        directory=directory,
        dataset_shape=[-1, 32, 32, self.IMAGE_DEPTH],
        train_size=24300,
        test_size=24300,
        num_train_classes=5,
        num_test_classes=5,
        num_classes=5)

    if self.IMAGE_DIM > self.SMALLNORB_SIZE:
      raise ValueError(
          'Image dim must be <= {}, got {}'.format(self.SMALLNORB_SIZE,
                                                   self.IMAGE_DIM))

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for smallNORB training data."""
    return self._dataset(
        'train',
        self._directory,
        'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
        'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
        preprocess)

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for smallNORB test data."""
    return self._dataset(
        'test',
        self._directory,
        'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
        'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
        preprocess)

  def _dataset(self, split, directory, images_file, labels_file, preprocess):
    """Download and parse smallNORB dataset."""

    images_file = self._download(directory, images_file)
    labels_file = self._download(directory, labels_file)

    self._check_image_file_header(images_file)
    self._check_labels_file_header(labels_file)

    def decode_image(image):
      image = tf.decode_raw(image, tf.uint8)
      image = tf.reshape(image, tf.stack(
          [self.SMALLNORB_DEPTH, self.SMALLNORB_SIZE, self.SMALLNORB_SIZE]))
      if self.IMAGE_DEPTH == 1:
        image = tf.expand_dims(image[0], 0)
      image = tf.transpose(image, [1, 2, 0])
      image = tf.cast(image, tf.float32)
      image = tf.image.resize_images(image, [self.IMAGE_DIM, self.IMAGE_DIM])
      return image

    def preprocess_image(image):
      """Distort the image using random brightness, contrast and crop."""
      if self.IMAGE_DIM == 48:
        distorted_dim = 32
      elif self.IMAGE_DIM == 32:
        distorted_dim = 22

      if preprocess:
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.random_crop(image, tf.stack(
            [distorted_dim, distorted_dim, self.IMAGE_DEPTH]))
      else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, distorted_dim, distorted_dim)

      image = tf.image.per_image_standardization(image)
      image.set_shape(self.shape[1:])
      image = tf.reshape(image, self.shape[1:])

      return image

    def decode_label(label):
      label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
      label = tf.to_int32(label[0])
      return label

    images = tf.data.FixedLengthRecordDataset(
        images_file,
        self.SMALLNORB_SIZE * self.SMALLNORB_SIZE * self.SMALLNORB_DEPTH,
        header_bytes=24)
    images = images.map(decode_image)
    images = images.map(preprocess_image)

    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 4, header_bytes=20).map(decode_label)

    return tf.data.Dataset.zip((images, labels))

  def _download(self, directory, filename):
    """Download (and unzip) a file from the smallNORB dataset if not already done."""
    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, filename)

    if tf.gfile.Exists(filepath):
      return filepath
    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)

    url = 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/' + (
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
    """Validate that filename corresponds to images for the smallNORB dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
      magic = self._read(f)
      self._read(f) # num_dims, unused
      self._read(f)  # num_images, unused
      depth = self._read(f)
      rows = self._read(f)
      cols = self._read(f)

      if magic != 85:
        raise ValueError(
            'Invalid magic number %d in smallNORB file %s' % (magic, f.name))

      if depth != self.SMALLNORB_DEPTH or rows != self.SMALLNORB_SIZE or (
          cols != self.SMALLNORB_SIZE):
        raise ValueError('Invalid smallNORB file %s: Expected %dx%dx%d images,'
                         ' found %dx%dx%d' % (
                             f.name, self.SMALLNORB_DEPTH, self.SMALLNORB_SIZE,
                             self.SMALLNORB_SIZE, depth, rows, cols))

  def _check_labels_file_header(self, filename):
    """Validate that filename corresponds to labels for the smallNORB dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
      magic = self._read(f)
      self._read(f) # num_dims, unused
      self._read(f)  # num_images, unused
      self._read(f)  # ignored integer
      self._read(f)  # ignored integer

      if magic != 84:
        raise ValueError(
            'Invalid magic number %d in smallNORB file %s' % (magic, f.name))

  def _read(self, bytestream, read_bytes=4, dtype=np.uint8):
    """Read 4 bytes from bytestream as an unsigned 8-bit integer."""
    dt = np.dtype(dtype).newbyteorder('>')
    return np.frombuffer(bytestream.read(read_bytes), dtype=dt)[0]
