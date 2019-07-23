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

"""Dataset base class."""

import abc


class Dataset:
  """Dataset base class based on tf.data."""

  def __init__(self, name, directory, dataset_shape,
               train_size, test_size, num_train_classes, num_test_classes,
               num_classes):
    self._name = name
    self._directory = directory
    self._dataset_shape = dataset_shape
    self._train_size = train_size
    self._test_size = test_size
    self._num_classes = num_classes
    self._num_train_classes = num_train_classes
    self._num_test_classes = num_test_classes

  @abc.abstractmethod
  def get_train(self, preprocess=False):
    """tf.data.Dataset object for training data."""
    raise NotImplementedError('Not implemented')

  @abc.abstractmethod
  def get_test(self, preprocess=False):
    """tf.data.Dataset object for test data."""
    raise NotImplementedError('Not implemented')

  def get_classes_by_superclass(self, superclass, proportion=1.0):
    pass

  @property
  def name(self):
    return self._name

  @property
  def shape(self):
    return self._dataset_shape

  @property
  def train_size(self):
    return self._train_size

  @property
  def test_size(self):
    return self._test_size

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def num_train_classes(self):
    return self._num_train_classes

  @property
  def num_test_classes(self):
    return self._num_test_classes

  def _calculate_epoch(self, dataset_size, batch_size, current_batch):
    return int((current_batch * batch_size) // dataset_size)

  def get_training_epoch(self, batch_size, batch):
    return self._calculate_epoch(self._train_size, batch_size, batch)

  def get_test_epoch(self, batch_size, batch):
    return self._calculate_epoch(self._test_size, batch_size, batch)
