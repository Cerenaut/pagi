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

"""ConvAutoencoderComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging

import numpy as np
import tensorflow as tf

from pagi.components.autoencoder_component import AutoencoderComponent
from pagi.utils.layer_utils import activation_fn
from pagi.utils.np_utils import np_write_filters


class ConvAutoencoderComponent(AutoencoderComponent):
  """
  Convolutional Autoencoder with tied weights (and untied biases).
  It has the feature that you can mask the hidden layer, to apply arbitrary
  sparsening.
  """

  def __init__(self):
    super().__init__()

    self._norm_filters = False

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = AutoencoderComponent.default_hparams()
    hparams.set_hparam('filters', 32)
    hparams.add_hparam('filters_field_width', 6)
    hparams.add_hparam('filters_field_height', 6)
    hparams.add_hparam('filters_field_stride', 3)
    return hparams

  @staticmethod
  def get_encoding_shape_4d(input_shape, hparams):
    return ConvAutoencoderComponent.get_convolved_shape(input_shape, hparams.filters_field_height,
                                                        hparams.filters_field_width,
                                                        hparams.filters_field_stride, hparams.filters,
                                                        padding='SAME')

  def _create_encoding_shape_4d(self, input_shape):
    return ConvAutoencoderComponent.get_encoding_shape_4d(input_shape, self._hparams)

  @staticmethod
  def get_convolved_shape(input_shape, field_height, field_width, field_stride, filters, padding='SAME'):
    """Compute convolved shape after padding with VALID or SAME."""
    batch_size = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]

    # VALID
    if padding == 'VALID':
      h = int((input_height - field_height) / field_stride + 1)
      w = int((input_width - field_width) / field_stride + 1)

    # SAME
    elif padding == 'SAME':
      h = math.ceil(float(input_height) / float(field_stride))
      w = math.ceil(float(input_width) / float(field_stride))

    # Useful:
    #print( "input_ w/h", input_width, input_height)
    #print( "field w/h", input_width, input_height)
    #print( "w/h", w, h)

    convolved_shape = [batch_size, h, w, filters]
    return convolved_shape

  def _build_encoding(self, input_tensor, mask_pl=None):
    """Build the encoder"""
    batch_input_shape = self._input_shape

    kernel_size = [
        self._hparams.filters_field_width, self._hparams.filters_field_height
    ]

    self._strides = [
        1, self._hparams.filters_field_stride, self._hparams.filters_field_stride, 1
    ]

    # Input reshape: Ensure 3D input for convolutional processing
    # -----------------------------------------------------------------
    self._input = tf.reshape(input_tensor, batch_input_shape, name='input')
    logging.debug(self._input)

    # Encoding
    # -----------------------------------------------------------------
    conv_filter_shape = [
        kernel_size[0],       # field w
        kernel_size[1],       # field h
        self._input_shape[3], # input depth
        self._hparams.filters # number of filters
    ]

    # Initialise weights and bias for the filter
    self._weights = tf.get_variable(
        shape=conv_filter_shape,
        initializer=tf.truncated_normal_initializer(stddev=0.03), name='weights')
    self._bias_encoding = tf.get_variable(
        shape=[self._hparams.filters],
        initializer=tf.zeros_initializer, name='bias_encoding')
    self._bias_decoding = tf.get_variable(
        shape=self._input_shape[1:],
        initializer=tf.zeros_initializer, name='bias_decoding')

    logging.debug(self._weights)
    logging.debug(self._bias_encoding)
    logging.debug(self._bias_decoding)

    # Setup the convolutional layer operation
    # Note: The first kernel is centered at the origin, not aligned to
    # it by its origin.
    convolved = tf.nn.conv2d(self._input, self._weights, self._strides, padding='SAME', name='convolved') # zero padded
    logging.debug(convolved)

    # Bias
    convolved_biased = tf.nn.bias_add(convolved, self._bias_encoding, name='convolved_biased')
    logging.debug(convolved_biased)

    # Nonlinearity
    # -----------------------------------------------------------------
    hidden_transfer, _ = activation_fn(convolved_biased, self._hparams.encoder_nonlinearity)

    # External masking
    # -----------------------------------------------------------------
    mask_shape = hidden_transfer.get_shape().as_list()
    mask_shape[0] = self._hparams.batch_size

    mask_pl = self._dual.add('mask', shape=mask_shape, default_value=1.0).add_pl()
    hidden_masked = tf.multiply(hidden_transfer, mask_pl)
    return hidden_masked, hidden_masked

  def _build_decoding(self, hidden_name, decoding_shape, filtered):  # pylint: disable=W0613
    """Build the decoder"""

    # Decoding: Reconstruction of the input
    # -----------------------------------------------------------------
    deconv_strides = self._strides
    deconv_shape = self._input_shape
    deconv_shape[0] = self._hparams.batch_size

    deconvolved = tf.nn.conv2d_transpose(
        filtered, self._weights, output_shape=deconv_shape,
        strides=deconv_strides, padding='SAME', name='deconvolved')
    logging.debug(deconvolved)

    # Reconstruction of the input, in 3d
    deconvolved_biased = tf.add(deconvolved, self._bias_decoding, name='deconvolved_biased')
    logging.debug(deconvolved_biased)

    # TODO: If adding bias, make it only 1 per conv location rather than 1 per pixel.

    # Reconstruction of the input, batch x 1D
    decoding_transfer, _ = activation_fn(deconvolved_biased, self._hparams.decoder_nonlinearity)
    decoding_reshape = tf.reshape(decoding_transfer, self._input_shape, name='decoding_reshape')
    logging.debug(decoding_reshape)
    return decoding_reshape

  def write_filters(self, session, folder=None):
    """Write the learned filters to disk."""
    # Original shape:  fh, fw, fd, filter
    # Transpose shape: filter, fh, fw, fd
    # Re shape:        filter, fh, fw * fd

    weights_values = session.run(self._weights)
    weights_shape = np.shape(weights_values)
    logging.debug('Weights shape: %s', weights_shape)

    field_depth = weights_shape[2]
    filters = weights_shape[3]
    field_width = self._hparams.filters_field_width * field_depth
    field_height = self._hparams.filters_field_height

    weights_transpose = np.transpose(weights_values, axes=[3, 0, 1, 2])
    logging.debug('Weights transpose shape: %s', weights_transpose.shape)
    weights_reshape = np.reshape(weights_transpose, [filters, field_height, field_width * field_depth])
    logging.debug('Weights re shape: %s', weights_reshape.shape)

    file_name = "filters_" + self.name + ".png"
    if folder is not None and folder != "":
      file_name = folder + '/' + file_name

    np_write_filters(weights_reshape, [field_height, field_width], file_name)

  def get_filters(self, session):
    weights_values = session.run(self._weights)
    return weights_values

  def set_norm_filters(self, val):
    self._norm_filters = val

  # TODO: Needs re-work, doesn't actually work since its a method not an attribute
  def _weights(self):
    """
    Norm the weights.

    Weights shape = [b, h, w, z]
    We want each filter to have sum 1 when in encoding
    """

    def norm_weights(filters):
      mute_dbug = False
      filters = tf_print(filters, "filters: ", mute=mute_dbug)
      frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(filters), axis=[1, 2, 3], keepdims=True))   # [b, z]
      frobenius_norm = tf_print(frobenius_norm, "filter sums: ", mute=mute_dbug)
      w = filters / frobenius_norm
      w = tf_print(w, "weights normed: ", mute=mute_dbug)
      return w

    weights = tf.cond(tf.logical_and(tf.equal(self._batch_type, 'encoding'),
                                     tf.equal(self._norm_filters, True)),
                      lambda: norm_weights(self._weights),
                      lambda: self._weights)

    return weights
