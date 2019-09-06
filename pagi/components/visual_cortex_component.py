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

"""VisualCortexComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from pagi.components.composite_component import CompositeComponent
from pagi.components.sparse_conv_maxpool import SparseConvAutoencoderMaxPoolComponent
from pagi.components.sparse_conv_autoencoder_component import SparseConvAutoencoderComponent


class VisualCortexComponent(CompositeComponent):
  """Multi-layered Visual Cortex (VC) architecture using Sparse Conv AEs."""

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = tf.contrib.training.HParams()
    component_hparams = SparseConvAutoencoderMaxPoolComponent.default_hparams()
    stack_hparams = ['num_layers', 'batch_size', 'sum_norm']

    for key, value in component_hparams.values().items():
      if key not in stack_hparams:
        hparams.add_hparam(key, [value])

    hparams.add_hparam('num_layers', 1)
    hparams.add_hparam('batch_size', component_hparams.batch_size)
    hparams.add_hparam('sum_norm', [-1])
    hparams.add_hparam('output_features', 'output')

    return hparams

  def get_loss(self):
    return self.get_sub_component(self._hparams.output_features).get_loss()

  def get_inputs(self):
    vc0 = self._sub_components['vc0']
    inputs = vc0.get_inputs()
    return inputs

  def get_output(self):
    return self.get_sub_component(self._hparams.output_features).get_encoding()

  def use_sum_norm(self, layer_idx):
    return self._hparams.sum_norm[layer_idx] != -1 and self._hparams.sum_norm[layer_idx] > 0

  def use_max_pool(self):
    return 'use_max_pool' in self._hparams

  def build(self, input_values, input_shape, hparams, name='vc'):
    """Initializes the model parameters.

    Args:
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        :param input_values:
        :param input_shape:
        :param hparams:
        :param name:
    """

    self.name = name
    self._hparams = hparams

    input_area = np.prod(input_shape[1:])

    logging.debug('Input Shape: %s', input_shape)
    logging.debug('Input Area: %s', input_area)

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

      input_values_next = input_values
      input_shape_next = input_shape

      # Build Visual Cortex (VC) with a Convolutional Sparse Autoencoder
      # ---------------------------------------------------------------------
      for i in range(self._hparams.num_layers):
        layer_name = self.get_layer_name(i)

        if self.use_max_pool():
          vc = SparseConvAutoencoderMaxPoolComponent()
        else:
          vc = SparseConvAutoencoderComponent()

        self._add_sub_component(vc, layer_name)

        # Build layer hparams
        if self.use_max_pool():
          hparams_vc = SparseConvAutoencoderMaxPoolComponent.default_hparams()
        else:
          hparams_vc = SparseConvAutoencoderComponent.default_hparams()
        hparams_vc = self.get_layer_hparams(i, hparams_vc)

        vc.build(input_values_next, input_shape_next, hparams_vc, layer_name)
        logging.info('%s / input = %s', layer_name, input_values_next)

        # Update 'next' value for VC+1
        input_values_next = vc.get_encoding_op()
        logging.info('%s / encoding = %s', layer_name, input_values_next)

        if self.use_max_pool():
          # Note: in 'training' mode, the usual get_encoding_op is correct: it has been pooled, and size can be decoded

          # Use pooled -> unpooled encoding, to pool but have appropriate size for decoding for viz, from higher levels
          if hparams_vc.use_max_pool in ['encoding', 'encoding_unpooled']:
            input_values_next = vc.get_encoding_unpooled_op()
            logging.info('%s / pool adjusted encoding (unpooled) = %s', layer_name, input_values_next)

          # Use pooled encoding directly. To decode from higher levels, you must unpool at a later stage in your workflows
          elif hparams_vc.use_max_pool == 'encoding_pooled':
            input_values_next = vc.get_encoding_pooled_op()
            logging.info('%s / pooled encoding = %s', layer_name, input_values_next)

        if self.use_sum_norm(i):
          logging.info('Using sum norm at layer %s with k=%s', i, self._hparams.sum_norm[i])
          input_values_next = tf_utils.tf_normalize_to_k(input_values_next, k=self._hparams.sum_norm[i], axis=[1, 2, 3])

        # Update 'next' shape for VC+1
        input_shape_next = [-1] + input_values_next.get_shape().as_list()[1:]

        self._output = input_values_next

    self.reset()

  def get_encoding_op(self, name='output'):
    vc = self.get_sub_component(name)
    return vc.get_encoding_op()

  def get_encoding(self, name='output'):
    vc = self.get_sub_component(name)
    return vc.get_encoding()

  def get_encoding_unpooled_op(self, name='output'):
    vc = self.get_sub_component(name)
    return vc.get_encoding_unpooled_op()

  def get_encoding_unpooled(self, name='output'):
    vc = self.get_sub_component(name)
    return vc.get_encoding_unpooled()

  def get_decoding_op(self, name='output'):
    vc = self.get_sub_component(name)
    return vc.get_decoding_op()

  def get_decoding(self, name='output'):
    vc = self.get_sub_component(name)
    return vc.get_decoding()

  def get_features(self, batch_type='training'):
    """
    The output of the component is taken from one of the subcomponents, depending on hparams.
    """
    del batch_type
    comp = self.get_sub_component(self._hparams.output_features)
    features = comp.get_features()
    return features
