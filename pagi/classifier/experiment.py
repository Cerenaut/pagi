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

"""Experiment class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import h5py

class Experiment(object):
  """
  The Experiment class represents a generic experiment that contains any
  set of features and labels. It includes options to export and import this
  data for later use.
  """

  def __init__(self, dataset, features=None, labels=None, num_classes=None, images=None):
    """
    Initializes the instance of the Experiment class.

    Args:
        dataset: The name of the dataset, e.g. mnist.
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing a label for each sample.
        num_classes: An integer, the number of unique classes.
    """
    self._dataset = dataset
    self._num_classes = num_classes

    self._features = features
    self._labels = labels
    self._images = images

  @property
  def features(self):
    """Returns the experiment's features."""
    return self._features

  @property
  def labels(self):
    """Returns the experiment's labels."""
    return self._labels

  @property
  def images(self):
    """Returns the experiment's images."""
    return self._images

  def get_data(self):
    """Returns the experiment's features, labels and images."""
    return [self._features, self._labels, self._images]

  def import_data(self, filepath):
    """
    Imports the experiment data from the specified H5Py file.

    Args:
        filepath: The path to the h5py compressed file.
    """
    with h5py.File(filepath, 'r') as hf:
      logging.debug('Reading features...')
      self._features = hf['features'][:]

      logging.debug('Reading labels...')
      self._labels = hf['labels'][:]

      if 'images' in hf:
        logging.debug('Reading images...')
        images = hf['images']
        if images is not None:
          self._images = images[:]
          logging.debug('Read images OK.')

  def export_data(self, output_dir='/tmp', filename=None):
    """
    Exports the experiment's data to disk using h5py.

    Args:
        output_dir: The location of the exported data file.
        filename: An override to allow setting a custom filename.
    """
    output_filename = 'output_%s.h5' % (self._dataset)
    output_directory = os.path.join(output_dir, 'experiment')

    # Override filename
    if filename is not None:
      output_filename = filename + '.h5'

    if not os.path.exists(output_directory):
      os.makedirs(output_directory)

    with h5py.File(
        os.path.join(output_directory, output_filename), 'w') as hf:
      hf.create_dataset('features', data=self._features, compression='lzf')
      hf.create_dataset('labels', data=self._labels, compression='lzf')
      if self._images is not None:
        logging.debug('Writing images...')
        hf.create_dataset('images', data=self._images, compression='lzf')
        logging.debug('Wrote images OK.')
