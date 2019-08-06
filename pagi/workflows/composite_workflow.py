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

"""CompositeWorkflow class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils
from pagi.workflows.workflow import Workflow

class CompositeWorkflow(Workflow):
  """The base workflow for working with composite components."""

  # TEST_DECODES = []

  # def _init_test_decodes(self):
  #   sub_component_keys = list(self._component.get_sub_components().keys())

  #   for i in range(len(sub_component_keys) - 1, 0, -1):
  #     self.TEST_DECODES.append((sub_component_keys[i], sub_component_keys[i-1])) # (name, decoder)

  #   self._num_repeats = len(self.TEST_DECODES) + 1

  #   return self.TEST_DECODES

  # def _setup_component(self):
  #   """Setup the component"""
  #   super()._setup_component()

  #   self._init_test_decodes()

  #   self._build_test_decodes_summaries()

  # def _add_composite_decodes(self, name, composite_decoder):
  #   """Calculate the number of decoders required. This takes into account any nested components."""
  #   num_decoders = 1

  #   try:
  #     num_sub_components = len(self._component.get_sub_component(composite_decoder).get_sub_components())

  #     if num_sub_components > 1:
  #       num_decoders = num_sub_components
  #   except AttributeError:
  #     pass

  #   for _ in range(num_decoders):
  #     self.TEST_DECODES.append((name, composite_decoder))

  # def _build_test_decodes_summaries(self):
  #   for _, (name, decoder) in enumerate(self.TEST_DECODES):
  #     self._component.build_secondary_decoding_summaries(name, decoder)

  def _setup_train_batch_types(self):
    sub_components = self._component.get_sub_components()

    batch_types = {}
    for key in sub_components.keys():
      name = self._component.name + '/' + key
      batch_types[name] = 'training'

    if self._checkpoint_opts['checkpoint_frozen_scope']:
      for key in self._checkpoint_opts['checkpoint_frozen_scope'].split(','):
        key = key.lstrip().rstrip()
        batch_types[key] = 'encoding'

    return batch_types

  def _decoder(self, batch, decoding_name, component_name, encoding, feed_dict, summarise=True):
    """
    Decode a given encoding using the specified component's trained decoder.
    Note: batch is ignored if summaries is False
    Example of decoding at multi-layered component:
      Input -> C (C_0 -> C_1) -> B -> A -> Output
      A & B are standard components, while C is a composite component (i.e. with sub-components)
      1. A is first decoded at B
      2. B is then decoded at C_1
      3. C_1 is finally decoded at C_0

    Args:
      batch: The number of the current batch/step
      decoding_name: The name of the component to decode
      component_name: The name of the component to decode with
      encoding: The encoding of the component to decode
      feed_dict: The constructed feed_dict (containing dataset handle, etc.)
      summarise: Creates decoding image summaries for TensorBoard
    """
    batch_type = 'secondary_decoding'
    has_sub_components = False

    sub_duals = {
        component_name: self._component.get_dual(component_name)
    }
    sub_components = {
        component_name: self._component.get_sub_component(component_name)
    }

    # Try to get dual and sub-components from the composite component
    # Otherwise, assume its not a composite component
    try:
      sub_components = sub_components[component_name].get_sub_components()
      # has_sub_components = True
      sub_duals = {}
      for k, v in sub_components.items():
        sub_duals[k] = v.get_dual()
    except AttributeError:
      pass

    # List of decoders, in reverse order
    decoders = list(sub_components.keys())
    decoders.reverse()

    # If A is a composite component with 3 layers, then 'decoders' (after reversing)
    # would be = [a/layer-3, a/layer-2, a/layer-1]
    # If the 'decoding_name' is in the decoder stack, e.g. if we try to decode layer-2 specifically
    # then remove the layers above (e.g. layer-3)
    if decoding_name in decoders:
      idx = decoders.index(decoding_name)
      decoders = decoders[idx:]

    decodings = []
    if decoding_name != component_name and decoding_name not in decoders:
      decodings.append(decoding_name)
    decodings += decoders

    for decoding, decoder in zip(decodings, decoders):
      summary = tf.Summary()

      dual = sub_duals[decoder]
      sub_component = sub_components[decoder]

      # Generate summary tag name
      summary_name = decodings[0]
      if has_sub_components:
        summary_name += '_' + decoder
      summary_tag = sub_component.name + '-summaries/' + batch_type + '/' + summary_name

      logging.debug('decoding %s using %s, shown at %s', decoding, sub_component.name, summary_tag)

      # Get placeholders
      secondary_decoding_input_pl = dual.get('secondary_decoding_input').get_pl()
      batch_type_pl = dual.get('batch_type').get_pl()

      secondary_decoding_feed_dict = feed_dict.copy()
      secondary_decoding_feed_dict.update({
          secondary_decoding_input_pl: encoding,
          batch_type_pl: batch_type
      })

      secondary_decoding_fetches = {}
      sub_component.add_fetches(secondary_decoding_fetches, batch_type)
      secondary_decoding_fetched = self._session.run(secondary_decoding_fetches, feed_dict=secondary_decoding_feed_dict)
      sub_component.set_fetches(secondary_decoding_fetched, batch_type)

      encoding = dual.get_values('decoding')

      # Summaries
      if summarise:
        summary_input_shape = image_utils.get_image_summary_shape(sub_component._input_shape)  # pylint: disable=protected-access
        summary_encoding = np.reshape(encoding, summary_input_shape)

        image_utils.arbitrary_image_summary(summary, summary_encoding,
                                            name=summary_tag)

        self._writer.add_summary(summary, batch)
        self._writer.flush()

    return encoding
