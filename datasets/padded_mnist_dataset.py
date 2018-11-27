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

"""Padded MNIST dataset using the tf.data module."""

import numpy as np
import tensorflow as tf

from utils.image_utils import pad_image, shift_image
from datasets.mnist_dataset import MNISTDataset

class PaddedMNISTDataset(MNISTDataset):
  """Padded version of the MNIST Dataset for affNIST generalization."""

  PAD = 6
  SHIFT = 6

  def __init__(self, directory):
    super(PaddedMNISTDataset, self).__init__(directory)

    self._dataset_shape = [-1, 40, 40, 1]

  def preprocess(self, image, preprocess):
    processed_image = pad_image(image, self.PAD)
    if preprocess:
      processed_image = shift_image(processed_image, self.SHIFT)
    return processed_image
