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

"""Small Omniglot dataset using the tf.data module."""

from pagi.datasets.omniglot_dataset import OmniglotDataset

class SmallOmniglotDataset(OmniglotDataset):
  """Smaller version of the Omniglot Dataset based on tf.data."""

  def __init__(self, directory):
    super(SmallOmniglotDataset, self).__init__(directory)

    self._train_size = 2720
    self._test_size = 3120
    self._num_train_classes = 136
    self._num_test_classes = 156
    self._num_classes = 292

  def get_train(self):
    """tf.data.Dataset object for small Omniglot training data."""
    return self._dataset(self._directory, 'images_background_small1')

  def get_test(self):
    """tf.data.Dataset object for small Omniglot test data."""
    return self._dataset(self._directory, 'images_background_small2')
