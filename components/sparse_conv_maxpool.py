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

"""SparseConvAutoencoderMaxPoolComponent class."""

import numpy as np
import tensorflow as tf
import logging

from components.conv_autoencoder_component import ConvAutoencoderComponent
from components.summarize_levels import SummarizeLevels
from utils import image_utils
from utils import layer_utils

from components.sparse_conv_autoencoder_component import SparseConvAutoencoderComponent

class SparseConvAutoencoderMaxPoolComponent(SparseConvAutoencoderComponent):
  """
  Sparse Convolutional Autoencoder with Max Pooling.

  Max pooling is controlled with hparam 'use_max_pooling'.
  use_max_pooling == 'encoding'
    - do not do pooling for training, but pool/unpool the encoding and make them available via an op

  use_max_pooling == 'training'
    - pool and unpool the encoding in recon path, and hence used for training
    - the 'encoding' op returns a pooled encoding
    - also sets separate ops for each version of encoding: pooled / unpooled.

  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = SparseConvAutoencoderComponent.default_hparams()
    hparams.add_hparam('use_unpooling', 'fixed')  # fixed, argmax
    hparams.add_hparam('use_max_pool', 'none')  # none, encoding, training

    hparams.add_hparam('pool_size', 2)
    hparams.add_hparam('pool_strides', 2)
    return hparams

  def __init__(self):
    super().__init__()

    self._pre_pooled_shape = None
    self._crop_unpooled = True

  def _create_encoding_shape_4d(self, input_shape):

    padding = 'SAME'

    encoding_shape = ConvAutoencoderComponent.get_convolved_shape(input_shape, self._hparams.filters_field_height,
                                                                  self._hparams.filters_field_width,
                                                                  self._hparams.filters_field_stride,
                                                                  self._hparams.filters, padding=padding)

    if self._hparams.use_max_pool and self._hparams.use_max_pool == 'training':
      if padding == 'SAME':
        encoding_shape[1] = int(np.ceil(encoding_shape[1] / self._hparams.pool_strides))
        encoding_shape[2] = int(np.ceil(encoding_shape[2] / self._hparams.pool_strides))
      elif padding == 'VALID':
        encoding_shape[1] = int(np.ceil(
          ((encoding_shape[1] - self._hparams.pool_size) / self._hparams.pool_strides) + 1))
        encoding_shape[2] = int(np.ceil(
          ((encoding_shape[2] - self._hparams.pool_size) / self._hparams.pool_strides) + 1))
      else:
        raise RuntimeError('Padding type unrecognized')

    return encoding_shape

  def get_encoding_unpooled(self):
    return self._dual.get_values('encoding_unpooled')

  def get_encoding_pooled_op(self):
    return self._dual.get_op('encoding_pooled')

  def get_encoding_unpooled_op(self):
    return self._dual.get_op('encoding_unpooled')

  def _build(self):
    """
    Build the subgraph for this component.
    If max_pool == 'encoding', apply pooling/unpooling, and set ops for later retrieval
    """
    super()._build()

    if self._hparams.use_max_pool == 'encoding':
      encoding = self._dual.get_op('encoding')        # note: after the stop gradient
      pooled, unpooled = self.pool_unpool(encoding)

      self._dual.set_op('encoding_pooled', pooled)
      self._dual.set_op('encoding_unpooled', unpooled)

  def _build_filtering(self, training_encoding, testing_encoding):
    """
    Returns filtered (sparse) versions of training and testing encoding (hidden state).
    If max_pool == 'training', apply pooling to both
    """

    training_filtered, testing_filtered = super()._build_filtering(training_encoding, testing_encoding)

    if self._hparams.use_max_pool != 'training':
      return training_filtered, testing_filtered

    self._pre_pooled_shape = training_filtered.get_shape().as_list()

    pooled_training_filtered, _ = self.pool(training_filtered)
    pooled_testing_filtered, _ = self.pool(testing_filtered)

    self._dual.set_op('encoding_pooled', pooled_testing_filtered)   # !!! Before stop gradient

    return pooled_training_filtered, pooled_testing_filtered

  def _build_decoding(self, hidden_name, decoding_shape, filtered):
    if self._hparams.use_max_pool in ['training']:
      unpooled_filtered = self.unpool(filtered, pre_pooled_shape=self._pre_pooled_shape)
      self._dual.set_op('encoding_unpooled', unpooled_filtered)

      # If this option is on, the encoder will be outputting a pooled encoding so we need to unpool prior to decoding
      filtered = unpooled_filtered

    decoded = super()._build_decoding(hidden_name, decoding_shape, filtered)
    return decoded

  def add_fetches(self, fetches, batch_type='training'):
    super().add_fetches(fetches, batch_type)

    if self._hparams.use_max_pool in ['training', 'encoding']:
      fetches[self._name]['encoding_pooled'] = self.get_encoding_pooled_op()
      fetches[self._name]['encoding_unpooled'] = self.get_encoding_unpooled_op()

  def set_fetches(self, fetched, batch_type='training'):
    super().set_fetches(fetched, batch_type)

    names = []
    if self._hparams.use_max_pool in ['training', 'encoding']:
      names = ['encoding_pooled', 'encoding_unpooled']
    self._dual.set_fetches(fetched, names)

  def _build_summaries(self):

    summaries = []
    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      return summaries

    summaries = super()._build_summaries()

    max_outputs = self._hparams.max_outputs

    if self.get_encoding_pooled_op() is not None:
      encoding_pooled = self.get_encoding_pooled_op()
      encoding_pooled_shape = encoding_pooled.get_shape().as_list()
      encoding_pooled_volume = np.prod(encoding_pooled_shape[1:])
      encoding_pooled_square_image_shape, _ = image_utils.square_image_shape_from_1d(encoding_pooled_volume)

      encoding_pooled_reshaped = tf.reshape(encoding_pooled, encoding_pooled_square_image_shape)
      summaries.append(tf.summary.image('encoding_pooled', encoding_pooled_reshaped, max_outputs=max_outputs))

    if self.get_encoding_unpooled_op() is not None:
      encoding_unpooled = self.get_encoding_unpooled_op()
      encoding_unpooled_shape = encoding_unpooled.get_shape().as_list()
      encoding_unpooled_volume = np.prod(encoding_unpooled_shape[1:])
      encoding_unpooled_square_image_shape, _ = image_utils.square_image_shape_from_1d(encoding_unpooled_volume)

      encoding_unpooled_reshaped = tf.reshape(encoding_unpooled, encoding_unpooled_square_image_shape)
      summaries.append(tf.summary.image('encoding_unpooled', encoding_unpooled_reshaped, max_outputs=max_outputs))

    return summaries

  def pool_unpool(self, tensor):
    """
    Applies the max pooling operation on a `Tensor` followed by the unpooling
    operation.

    Crop the output so that it is the same size as input.
    Otherwise there are adverse downstream effects
    e.g. output size is used for decoding (visualisation) or
    de-convolving (training), and it doesn't work because the size is wrong.

    Returns both pooled and unpooled `Tensor`.
    """
    pool_size = self._hparams.pool_size
    pool_strides = self._hparams.pool_strides

    pooled, mask = self.pool(tensor, pool_size, pool_strides)
    unpooled = self.unpool(pooled, pool_size, pool_strides, mask, pre_pooled_shape=tensor.shape.as_list())

    return pooled, unpooled

  def pool(self, tensor, pool_size=None, pool_strides=None):
    """Perform the Max Pooling operation on a tensor."""
    if pool_size is None:
      pool_size = self._hparams.pool_size
    if pool_strides is None:
      pool_strides = self._hparams.pool_strides

    mask = None

    if self._hparams.use_unpooling == 'argmax':
      pooled, mask = layer_utils.max_pool_with_argmax(tensor,
                                                      pool_size,
                                                      pool_strides,
                                                      padding='SAME')
    else:
      pooled = tf.layers.max_pooling2d(tensor, pool_size, pool_strides, padding='SAME')

    return pooled, mask

  def unpool(self, pooled_tensor, pool_size=None, pool_strides=None, indices=None, pre_pooled_shape=None):
    """
    Performs unpooling on an existing pooled tensor.

    Note: unpool_with_argmax uses indices/scatter_nd, and currently
    does not work with batch_size > 1. Keeping it here as it might come in
    handy later if we ever need to restore the true max positions.
    """
    if pool_size is None:
      pool_size = self._hparams.pool_size
    if pool_strides is None:
      pool_strides = self._hparams.pool_strides

    if self._hparams.use_unpooling == 'argmax':
      raise RuntimeWarning('Unpool with argmax is currently not working. See comments in code.')

      unpooled = layer_utils.unpool_with_argmax(pooled_tensor, indices,
                                                ksize=[1, pool_size, pool_size, 1])

    elif self._hparams.use_unpooling == 'fixed':
      unpooled = layer_utils.unpool_with_fixed(pooled_tensor, pool_strides, mode='zeros')
    else:
      raise NotImplementedError('Unpooling method not supported: ' + str(self._hparams.use_unpooling))

    # crop the unpooled value to match dimensions with pre-pooled version
    if self._crop_unpooled and pre_pooled_shape is not None:

      height = pre_pooled_shape[1]
      width = pre_pooled_shape[2]

      unpooled = tf.image.crop_to_bounding_box(unpooled, 0, 0, height, width)

      unpooled_shape = unpooled.shape.as_list()
      up_height = unpooled_shape[1]
      up_width = unpooled_shape[2]

      if up_height > height or up_width > width:
        logging.warning("Un-Pooled volume was cropped, up-height %d, up-width: %d, height %d, width: %d.",
                        up_height,
                        up_width,
                        height,
                        width)

    return unpooled
