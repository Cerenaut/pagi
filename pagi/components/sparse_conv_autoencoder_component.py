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

"""SparseConvAutoencoderComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pagi.components.conv_autoencoder_component import ConvAutoencoderComponent
from pagi.utils.tf_utils import tf_build_top_k_mask_4d_op


class SparseConvAutoencoderComponent(ConvAutoencoderComponent):
  """
  Sparse Convolutional Autoencoder with tied weights (and untied biases).
  It has the feature that you can mask the hidden layer, to apply arbitrary
  sparsening.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = ConvAutoencoderComponent.default_hparams()
    hparams.add_hparam('sparsity', 5)
    hparams.add_hparam('sparsity_output_factor', 1.5)
    return hparams

  def _build_filtering(self, training_encoding, testing_encoding):
    """Build filtering/masking for specified encoding."""
    top_k_input = training_encoding
    top_k2_input = testing_encoding
    hidden_size = self._hparams.filters
    batch_size = self._hparams.batch_size

    encoding_shape = training_encoding.get_shape().as_list()
    h = encoding_shape[1]
    w = encoding_shape[2]

    # Find the "winners". The top k elements in each batch sample. this is
    # what top_k does.
    # ---------------------------------------------------------------------
    k = int(self._hparams.sparsity)
    k2 = int(self._hparams.sparsity * self._hparams.sparsity_output_factor)  # verified k_output=75

    top_k_mask = tf_build_top_k_mask_4d_op(top_k_input, k, batch_size, h, w, hidden_size)
    top_k2_mask = tf_build_top_k_mask_4d_op(top_k2_input, k2, batch_size, h, w, hidden_size)

    # Retrospectively add batch-sparsity per cell: pick the top-k (for now
    # k=1 only). TODO make this allow top N per batch.
    # ---------------------------------------------------------------------
    batch_max = tf.reduce_max(top_k_input, axis=[0, 1, 2], keepdims=True)  # input shape: batch,cells, output shape: cells
    batch_mask_bool = tf.greater_equal(top_k_input, batch_max) # inhibit cells (mask=0) until spike has decayed
    batch_mask = tf.to_float(batch_mask_bool) # i.e. 1s in each cell-batch where it's the most active per batch

    # Apply the 3 masks
    # ---------------------------------------------------------------------
    either_mask = tf.maximum(top_k_mask, batch_mask) # logical OR, i.e. top-k or top-1 per cell in batch
    training_filtered = training_encoding * either_mask # apply mask 3 to output 2
    testing_filtered = testing_encoding * top_k2_mask

    return training_filtered, testing_filtered
