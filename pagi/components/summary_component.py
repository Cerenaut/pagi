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

"""SummaryComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import shutil

import tensorflow as tf

from pagi.components.dual_component import DualComponent


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
      _, _, free = shutil.disk_usage(writer.get_logdir())  # total, used, free

      # TODO: Instead of waiting till it reaches 0, we can adjust the check to be
      # for a percentage of the total. Example: free > 0.10 * total (or something)
      if free == 0:
        raise OSError(errno.ENOSPC, 'No space left on device', writer.get_logdir())

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
    raise NotImplementedError('_build_summaries should be implemented in child components.')
