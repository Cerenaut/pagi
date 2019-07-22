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

"""AutoencoderComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf

from utils.dual import DualData
from utils.layer_utils import activation_fn
from utils.np_utils import np_write_filters
from utils.tf_utils import tf_build_stats_summaries
from utils import image_utils

d = dirname(dirname(dirname(abspath(__file__)))) # Each dirname goes up one
sys.path.append(d)
sys.path.append(os.path.join(d, 'classifier_component'))

from classifier_component.component import Component # pylint: disable=C0413
from components.dual_component import DualComponent


class SummaryComponent(DualComponent):
  """
  A Component with a systematic way of building summaries efficiently.
  """

  def __init__(self):
    super().__init__()

    self._summary_ops = {}
    self._summary_values = {}

  def build_summaries(self, batch_types=None, max_outputs=3, scope=None):
    """Builds all summaries."""

    # Default scope
    if not scope:
      name = self.name
      scope = name + '/summaries/'

    # Default list
    if batch_types is None:
      batch_types = self.get_batch_types()

    with tf.name_scope(scope):
      for batch_type in batch_types:
        with tf.name_scope(batch_type):
          summaries = self._build_summaries(batch_type, max_outputs)
          if summaries is not None:
            summary_op = tf.summary.merge(summaries)
            self._summary_ops[batch_type] = summary_op
            #self._summary_values[batch_type] = None  # Init

  def write_summaries(self, step, writer, batch_type='training'):
    if batch_type in self._summary_values.keys():
      summary_values = self._summary_values[batch_type]
      writer.add_summary(summary_values, step)
      writer.flush()

  def add_fetches(self, fetches, batch_type='training'):
    name = self.name
    if batch_type in self._summary_ops.keys():
      summary_op = self._summary_ops[batch_type]
      fetches[name + '-summaries'] = summary_op

  def set_fetches(self, fetched, batch_type='training'):
    name = self.name
    if batch_type in self._summary_ops.keys():
      summary_values = fetched[name + '-summaries']
      self._summary_values[batch_type] = summary_values

  def _build_summaries(self, batch_type, max_outputs=3):
    """Build summaries for this batch type. Can be same for all batch types."""
    return None
