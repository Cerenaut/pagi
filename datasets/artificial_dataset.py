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

"""Dataset generated with random process."""


from datasets.dataset import Dataset
import numpy as np
import tensorflow as tf


class ArtficialDataset(Dataset):
  """
  Artificially generated dataset.
  Cache train/test datasets, so if fetched again, don't need to regenerate
  And this guarantees reproducability
  """

  def __init__(self, directory):
    super(ArtficialDataset, self).__init__(
        name='artificial_dataset',
        directory=directory,
        dataset_shape=[-1, 38, 38, 1],  # this is arbitrary, same as DG for exp, in future make it settable
        train_size=100,
        test_size=100,
        num_train_classes=1,
        num_test_classes=1,
        num_classes=1)
    self._train = None
    self._test = None

  def get_train(self, preprocess=False):
    """tf.data.Dataset object for artificial training data."""
    if self._train is None:
      self._train = self._dataset(self.train_size)
    return self._train

  def get_test(self, preprocess=False):
    """tf.data.Dataset object for artificial test data."""
    if self._test is None:
      self._test = self._dataset(self.test_size)
    return self._test

  def _dataset(self, set_size):
    """Generate data."""

    sparsity = 0.9    # fractional sparsity e.g. 0.5 = 0.5 active,   0.2 = 0.8 active
    binary = True     # generate binary values(0, 1) or if false, floats between[0 and 1)
    min_value = -1    # the value for binary 'inactive', samples will be from (min_value, 1)

    sample_size = self.shape[1] * self.shape[2]
    length_sparse = int(sample_size * sparsity)

    if binary:
      proto_sample = np.ones(sample_size, dtype=np.float32)
    else:
      proto_sample = np.random.rand(sample_size)

    proto_sample[:length_sparse] = min_value

    images = np.zeros([set_size, self.shape[1], self.shape[2], self.shape[3]], dtype=np.float32)
    labels = []
    for i in range(set_size):
      sample = np.random.permutation(proto_sample)
      sample = np.resize(sample, (self.shape[1], self.shape[2], self.shape[3]))
      images[i] = sample
      labels.append(1)   # if not superclass, this must be an integer

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset






