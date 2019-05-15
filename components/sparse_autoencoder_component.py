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

"""SparseAutoencoderComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from components.autoencoder_component import AutoencoderComponent
from utils.tf_utils import tf_build_top_k_mask_op


class SparseAutoencoderComponent(AutoencoderComponent):
  """
  Sparse Autoencoder with tied weights (and untied biases).
  It has the feature that you can mask the hidden layer, to apply arbitrary
  sparsening.

  Derived from Makhzani & Frey's two k-Sparse Autoencoder papers.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = AutoencoderComponent.default_hparams()
    hparams.add_hparam('sparsity', 25)
    hparams.add_hparam('sparsity_output_factor', 3.0)

    # TODO restore frequency testing
    # hparams.add_hparam('frequency_learning_rate', 0.1)
    # hparams.add_hparam('frequency_update_interval', 30)

    return hparams

  def _build_filtering(self, training_encoding, testing_encoding):
    """Build the encoding filtering."""
    top_k_input = training_encoding
    top_k2_input = testing_encoding
    hidden_size = self._hparams.filters
    batch_size = self._hparams.batch_size

    # Find the "winners". The top k elements in each batch sample. this is
    # what top_k does.
    # ---------------------------------------------------------------------
    k = int(self._hparams.sparsity)
    top_k_mask = tf_build_top_k_mask_op(top_k_input, k, batch_size, hidden_size)
    #top_k_masked = top_k_input * top_k_mask

    # Retrospectively add batch-sparsity per cell: pick the top-k (for now
    # k=1 only). TODO make this allow top N per batch.
    # ---------------------------------------------------------------------
    batch_max = tf.reduce_max(top_k_input, axis=0, keepdims=True)  # input shape: batch,cells, output shape: cells
    batch_mask_bool = tf.greater_equal(top_k_input, batch_max)  # inhibit cells (mask=0) until spike has decayed
    batch_mask = tf.to_float(batch_mask_bool)  # i.e. 1s in each cell-batch where it's the most active per batch
    either_mask = tf.maximum(top_k_mask, batch_mask)  # logical OR, i.e. top-k or top-1 per cell in batch
    training_filtered = training_encoding * either_mask  # apply mask 3 to output 2

    # perform a higher density top-k for classification purposes.
    # This has been shown to be generally a good idea with sparse codes.
    # ---------------------------------------------------------------------
    k2 = int(self._hparams.sparsity * self._hparams.sparsity_output_factor)  # verified k_output=75
    top_k2_mask = tf_build_top_k_mask_op(top_k2_input, k2, batch_size, hidden_size)
    testing_filtered = testing_encoding * top_k2_mask

    return training_filtered, testing_filtered
