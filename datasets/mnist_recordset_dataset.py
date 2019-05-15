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

"""MNIST dataset read in from a TFRecordSet, using the tf.data module."""


from datasets.dataset import Dataset
from utils.data_utils import read_subset, generate_filenames


class MNISTRecordsetDataset(Dataset):
  """MNIST Dataset based on tf.data."""

  def __init__(self, directory):
    super(MNISTRecordsetDataset, self).__init__(
        name='mnist_recordset',
        directory=directory,
        dataset_shape=[-1, 28, 28, 1],
        train_size=500,
        test_size=500,
        num_train_classes=1,
        num_test_classes=1,
        num_classes=1)

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for MNIST training data."""
    return self._dataset(self._directory, 'mnist_subset.tfrecords')

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for MNIST test data."""
    return self._dataset(self._directory, 'mnist_subset.tfrecords')

  def _dataset(self, directory, data_file):
    """Load MNIST from a TFRecordset specified by 'filepath'."""

    filenames = generate_filenames(self.name, directory, data_file)
    dataset_shape = self.shape
    dataset = read_subset(filenames, dataset_shape)
    return dataset


