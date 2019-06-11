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

"""EpisodicComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging

import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf

from components.sparse_conv_autoencoder_component import SparseConvAutoencoderComponent
from utils.dual import DualData
from components.sparse_conv_maxpool import SparseConvAutoencoderMaxPoolComponent

# Supervised Components
d = dirname(dirname(dirname(abspath(__file__)))) # Each dirname goes up one
sys.path.append(d)
sys.path.append(os.path.join(d, 'classifier_component'))

from classifier_component.feature_detector import FeatureDetector  # pylint: disable=C0413


class VisualCortexComponent(FeatureDetector):
  """Multi-layered Visual Cortex (VC) architecture using Sparse Conv AEs."""

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    hparams = SparseConvAutoencoderMaxPoolComponent.default_hparams()

    # Ensure layer hparams are list-based
    for key, value in hparams.values().items():
      hparams.del_hparam(key)
      hparams.add_hparam(key, [value])

    # Define the VC-level hparams
    hparams.add_hparam('output_features', 'vc_n')
    hparams.add_hparam('num_layers', 1)

    return hparams

  def __init__(self):
    super(VisualCortexComponent, self).__init__()

    self._name = None
    self._hparams = None
    self._summary_op = None
    self._summary_result = None
    self._dual = None

    self._decoder_summaries = []
    self._sub_components = {}  # map {name, component}

    self._layered_hparams = None
    self._output = None

  @property
  def name(self):
    return self._name

  def get_inputs(self):
    vc0 = self._sub_components['vc0']
    inputs = vc0.get_inputs()
    return inputs

  def get_loss(self):
    """Define loss as the loss of the subcomponent selected for output features: using _hparam.output_features"""

    comp = self.get_sub_component(self._hparams.output_features)
    return comp.get_loss()

  def build(self, input_values, input_shape, hparams, name='vc'):
    """Initializes the model parameters.

    Args:
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        :param input_values:
        :param input_shape:
        :param hparams:
        :param name:
    """

    use_vcmaxpool_class = True

    self._name = name
    self._hparams = hparams
    self._dual = DualData(self._name)
    self._layered_hparams = self.build_layer_hparams()

    input_area = np.prod(input_shape[1:])

    logging.debug('Input Shape: %s', input_shape)
    logging.debug('Input Area: %s', input_area)

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

      input_values_next = input_values
      input_shape_next = input_shape

      # Build Visual Cortex (VC) with a Convolutional Sparse Autoencoder
      # ---------------------------------------------------------------------
      for i in range(self._hparams.num_layers):
        layer_name = self.get_layer_name(i)

        if use_vcmaxpool_class:
          vc = SparseConvAutoencoderMaxPoolComponent()
        else:
          vc = SparseConvAutoencoderComponent()

        self._add_sub_component(vc, layer_name)

        hparams_override = self._layered_hparams[layer_name]
        if use_vcmaxpool_class:
          hparams_vc = SparseConvAutoencoderMaxPoolComponent.default_hparams()
        else:
          hparams_vc = SparseConvAutoencoderComponent.default_hparams()
        hparams_vc = hparams_vc.override_from_dict(hparams_override)

        print('vc layer: ', str(i), ' input', input_values_next)
        vc.build(input_values_next, input_shape_next, hparams_vc, layer_name)

        # Update 'next' value for VC+1
        input_values_next = vc.get_encoding_op()

        print('vc layer: ', str(i), ' encoding', input_values_next)

        if use_vcmaxpool_class and hparams_vc.use_max_pool == 'encoding':
          # Use pooled->unpooled encoding, to pool but have appropriate size for decoding for viz, from higher levels
          # Note: in 'training' mode, the usual get_encoding_op is correct: it has been pooled, and size can be decoded
          input_values_next = vc.get_encoding_unpooled_op()
          print('vc layer: ', str(i), ' pool adjusted encoding (unpooled)', input_values_next)

        # Update 'next' shape for VC+1
        input_shape_next = [-1] + input_values_next.get_shape().as_list()[1:]

        self._output = input_values_next

    self.reset()

  def get_layer_name(self, layer):
    return self.name + str(layer)

  def build_layer_hparams(self):
    """Parse list-based combined HParams to a separate layer-based HParams."""
    hparams = {}

    for i in range(self._hparams.num_layers):
      layer_hparams = {}
      layer_name = self.get_layer_name(i)

      for key, value in self._hparams.values().items():
        if isinstance(value, list):
          if len(value) < self._hparams.num_layers:
            layer_hparams[key] = value[0]
          else:
            layer_hparams[key] = value[i]

      hparams[layer_name] = layer_hparams

    return hparams

  def _add_sub_component(self, component, name):
    self._sub_components[name] = component

  def get_sub_components(self):
    return self._sub_components

  def get_sub_component(self, name):
    if name == 'vc_n':
      name = list(self._sub_components.keys())[-1]  # Get final VC
    if name in self._sub_components:
      return self._sub_components[name]
    return None

  def get_batch_type(self, name=None):
    """
    Return dic of batch types for each component (key is component)
    If component does not have a persistent batch type, then don't include in dictionary,
    assumption is that in that case, it doesn't have any effect.
    """
    if name is None:
      batch_types = []
      for c in self._sub_components:
        if hasattr(self._sub_components[c], 'get_batch_type'):
          batch_types.append(self._sub_components[c].get_batch_type())
      return max(set(batch_types), key=batch_types.count)

    return self._sub_components[name].get_batch_type()

  def get_dual(self, name=None):
    if name is None:
      return self._dual
    return self._sub_components[name].get_dual()

  def get_encoding_op(self, name='vc_n'):
    vc = self.get_sub_component(name)
    return vc.get_encoding_op()

  def get_encoding(self, name='vc_n'):
    vc = self.get_sub_component(name)
    return vc.get_encoding()

  def get_encoding_unpooled_op(self, name='vc_n'):
    vc = self.get_sub_component(name)
    return vc.get_encoding_unpooled_op()

  def get_encoding_unpooled(self, name='vc_n'):
    vc = self.get_sub_component(name)
    return vc.get_encoding_unpooled()

  def get_decoding_op(self, name='vc_n'):
    vc = self.get_sub_component(name)
    return vc.get_decoding_op()

  def get_decoding(self, name='vc_n'):
    vc = self.get_sub_component(name)
    return vc.get_decoding()

  def get_output(self):
    return self._output

  def get_features(self, batch_type='training'):
    """
    The output of the component is taken from one of the subcomponents, depending on hparams.
    """
    comp = self.get_sub_component(self._hparams.output_features)
    features = comp.get_features()
    return features

  def reset(self):
    """Reset the trained/learned variables and all other state of the component to a new random state."""
    for c in self._sub_components.values():
      c.reset()

  def update_feed_dict(self, feed_dict, batch_type='training'):
    for _, comp in self._sub_components.items():
      comp.update_feed_dict(feed_dict, batch_type)

  def add_fetches(self, fetches, batch_type='training'):
    for _, comp in self._sub_components.items():
      comp.add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    for _, comp in self._sub_components.items():
      comp.set_fetches(fetched, batch_type)

  def build_summaries(self, batch_types=None, scope=None):
    """Build summaries."""
    if batch_types is None:
      batch_types = []

    for name, comp in self._sub_components.items():
      consolidate_graph_view = False

      scope = name + '-summaries'   # this is best for visualising images in summaries
      if consolidate_graph_view:
        scope = self._name + '/' + name + '/summaries/'

      comp.build_summaries(batch_types, scope=scope)

  def build_secondary_decoding_summaries(self, scope, name):
    """Builds secondary decoding summaries."""
    for decoder_name, comp in self._sub_components.items():
      scope = decoder_name + '-summaries/'
      if name != self.name:
        summary_name = name + '_' + decoder_name
        if summary_name not in self._decoder_summaries:
          comp.build_secondary_decoding_summaries(scope, summary_name)
          self._decoder_summaries.append(summary_name)
      else:
        for decoded_name in self._sub_components:
          summary_name = decoded_name + '_' + decoder_name
          if summary_name not in self._decoder_summaries:
            comp.build_secondary_decoding_summaries(scope, summary_name)
            self._decoder_summaries.append(summary_name)

  def write_summaries(self, step, writer, batch_type='training'):
    for _, comp in self._sub_components.items():
      comp.write_summaries(step, writer, batch_type)

  def write_filters(self, session, folder):
    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      for _, comp in self._sub_components.items():
        if hasattr(comp, 'write_filters'):
          comp.write_filters(session, folder)

  def _select_batch_type(self, batch_type, name, as_list=False):
    name = self._name + '/' + name
    if isinstance(batch_type, dict):
      if name in batch_type:
        batch_type = batch_type[name]
    if as_list and not isinstance(batch_type, list):
      batch_type = [batch_type]
    return batch_type
