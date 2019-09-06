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

"""CompositeComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pagi.components.summary_component import SummaryComponent


class CompositeComponent(SummaryComponent):
  """The base class for building composite components (i.e. containing sub components)."""

  def __init__(self):
    super().__init__()

    self._sub_components = {}  # map {name, component}
    self._consolidate_graph_view = False

  def _build_summaries(self, batch_type, max_outputs=3):
    pass

  def _select_batch_type(self, batch_type, name, as_list=False):
    name = self.name + '/' + name
    if isinstance(batch_type, dict):
      if name in batch_type:
        batch_type = batch_type[name]
    if as_list and not isinstance(batch_type, list):
      batch_type = [batch_type]
    return batch_type

  def _add_sub_component(self, component, name):
    self._sub_components[name] = component

  def get_sub_components(self):
    return self._sub_components

  def get_sub_component(self, name):
    if name == 'output':
      name = list(self._sub_components.keys())[-1]  # Get output component
    if name in self._sub_components:
      return self._sub_components[name]
    return None

  def get_dual(self, name=None):  # pylint: disable=arguments-differ
    if name is None:
      return self._dual
    return self.get_sub_component(name).get_dual()

  def get_layer_name(self, layer_idx):
    return self.name + '/layer-' + str(layer_idx + 1)

  def get_layer_hparams(self, layer_idx, layer_hparams):
    """Parse list-based combined HParams to a separate layer-based HParams."""
    global_hparams_dict = self._hparams.values()

    for key, _ in layer_hparams.values().items():
      if key in global_hparams_dict.keys():
        hparam_value = global_hparams_dict[key]
        if isinstance(global_hparams_dict[key], list):
          try:
            hparam_value = global_hparams_dict[key][layer_idx]
          except IndexError:
            hparam_value = global_hparams_dict[key][0]

        layer_hparams.set_hparam(key, hparam_value)

    return layer_hparams

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

  def update_feed_dict(self, feed_dict, batch_type='training'):
    for name, comp in self._sub_components.items():
      comp.update_feed_dict(feed_dict, self._select_batch_type(batch_type, name))

  def add_fetches(self, fetches, batch_type='training'):
    for name, comp in self._sub_components.items():
      comp.add_fetches(fetches, self._select_batch_type(batch_type, name))

  def set_fetches(self, fetched, batch_type='training'):
    for name, comp in self._sub_components.items():
      comp.set_fetches(fetched, self._select_batch_type(batch_type, name))

  def build_summaries(self, batch_types=None, max_outputs=3, scope=None):
    if batch_types is None:
      batch_types = []

    components = self._sub_components.copy()

    for name, comp in components.items():
      scope = name + '-summaries'   # this is best for visualising images in summaries
      if self._consolidate_graph_view:
        scope = self.name + '/' + name + '/summaries/'

      batch_type = self._select_batch_type(batch_types, name, as_list=True)

      comp.build_summaries(batch_type, scope=scope)

  def write_summaries(self, step, writer, batch_type='training'):
    for name, comp in self._sub_components.items():
      comp.write_summaries(step, writer, self._select_batch_type(batch_type, name))

  def write_filters(self, session, folder):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      for _, comp in self._sub_components.items():
        if hasattr(comp, 'write_filters'):
          comp.write_filters(session, folder)

  def reset(self):
    """Reset the trained/learned variables and all other state of the component to a new random state."""
    for c in self._sub_components.values():
      c.reset()
