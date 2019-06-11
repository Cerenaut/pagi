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

"""Omniglot dataset using the tf.data module."""

import os
import math
import zipfile
import tempfile
import logging

from random import shuffle
from six.moves import urllib

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset


class OmniglotDataset(Dataset):
  """Omniglot Dataset based on tf.data."""

  # Mapping of alphabets (superclasses) to characeters (classes) populated when dataset is loaded
  CLASS_MAP = {}

  def __init__(self, directory):
    super(OmniglotDataset, self).__init__(
        name='omniglot',
        directory=directory,
        dataset_shape=[-1, 105, 105, 1],
        train_size=19280,
        test_size=13180,
        num_train_classes=964,
        num_test_classes=659,
        num_classes=1623)

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for Omniglot training data."""
    return self._dataset(self._directory, 'images_background')

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for Omniglot test data."""
    return self._dataset(self._directory, 'images_evaluation')

  def get_classes_by_superclass(self, superclass, proportion=1.0):
    """
    Retrieves a proportion of classes belonging to a particular superclass, defaults to retrieving all classes
    i.e. proportion=1.0.

    Arguments:
      superclass: A list containing the names of superclasses, or a single name of a superclass.
      proportion: A float that indicates the proportion of sub-classes to retrieve (default=1.0)
    """
    if not self.CLASS_MAP:
      raise ValueError('Superclass to class mapping (CLASS_MAP) is not populated yet.')

    def filter_classes(classes, proportion, do_shuffle=True):
      """Filters the list of classes by retrieving a proportion of shuffled classes."""
      if do_shuffle:
        shuffle(classes)
      num_classes = math.ceil(len(classes) * float(proportion))
      return classes[:num_classes]

    classes = []
    if isinstance(superclass, list):
      for i in superclass:
        subclasses = filter_classes(self.CLASS_MAP[i], proportion)
        classes.extend(subclasses)
    else:
      classes = filter_classes(self.CLASS_MAP[superclass], proportion)

    return classes

  def _dataset(self, directory, images_file):
    """Download and parse Omniglot dataset."""

    images_folder = self._download(directory, images_file)

    filenames, labels = self._filenames_and_labels(images_folder)

    def parse_function(filename, label):
      """Read and parse the image from a filepath."""
      image_string = tf.read_file(filename)

      # Don't use tf.image.decode_image, or the output shape will be undefined
      image = tf.image.decode_jpeg(image_string, channels=self.shape[3])

      # This will convert to float values in [0, 1]
      image = tf.image.convert_image_dtype(image, tf.float32)

      # Resize image and flatten feature dimension
      image = tf.image.resize_images(image, [self.shape[1], self.shape[2]])
      image = tf.reshape(image, self._dataset_shape[1:])

      return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=4)

    return dataset

  def _download(self, directory, filename):
    """Download (and unzip) a file from the Omniglot dataset if not already done."""
    dirpath = os.path.join(directory, self.name)
    filepath = os.path.join(dirpath, filename)
    if tf.gfile.Exists(filepath):
      return filepath
    if not tf.gfile.Exists(dirpath):
      tf.gfile.MakeDirs(dirpath)

    url = 'https://github.com/brendenlake/omniglot/raw/master/python/' + (
        filename + '.zip')
    _, zipped_filepath = tempfile.mkstemp(suffix='.zip')
    logging.info('Downloading %s to %s', url, zipped_filepath)
    urllib.request.urlretrieve(url, zipped_filepath)

    zip_ref = zipfile.ZipFile(zipped_filepath, 'r')
    zip_ref.extractall(dirpath)
    zip_ref.close()

    os.remove(zipped_filepath)
    return filepath

  def _filenames_and_labels(self, image_folder):
    """Get the image filename and label for each Omniglot character."""
    # Compute list of characters (each is a folder full of images)
    character_folders = []
    for family in os.listdir(image_folder):
      if os.path.isdir(os.path.join(image_folder, family)):
        append_characters = False
        if family not in self.CLASS_MAP:
          self.CLASS_MAP[family] = []
          append_characters = True
        for character in os.listdir(os.path.join(image_folder, family)):
          character_folder = os.path.join(image_folder, family, character)
          if append_characters and os.path.isdir(character_folder):
            character_file = os.listdir(character_folder)[0]
            character_label = int(character_file.split('_')[0])
            self.CLASS_MAP[family].append(character_label)
          character_folders.append(character_folder)
      else:
        logging.warning('Path to alphabet is not a directory: %s', os.path.join(image_folder, family))

    # Count number of images
    num_images = 0
    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          num_images += 1

    # Put them in one big array, and one for labels
    #   A 4D uint8 numpy array [index, y, x, depth].
    idx = 0
    filename_arr = []
    label_arr = np.zeros([num_images], dtype=np.int32)

    for path in character_folders:
      if os.path.isdir(path):
        for file in os.listdir(path):
          filename_arr.append(os.path.join(path, file))
          label_arr[idx] = file.split('_')[0]
          idx += 1

    return filename_arr, label_arr

  def set_shape(self, height, width):
    self._dataset_shape[1] = height
    self._dataset_shape[2] = width
