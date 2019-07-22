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

"""MovingAverageSummaries class"""

import numpy as np
import tensorflow as tf

from pagi.utils.tf_utils import tf_get_summary_tag
from pagi.utils.tf_utils import tf_write_scalar_summary

class MovingAverageSummaries(object):
  """
  Manages the online calculation of accuracy statistics and summaries.
  Helper for Workflows that need this feature.
  """

  def __init__(self):
    self._summaries = {}

  def clear(self):
    for key,val in self._summaries.items():
      val.clear()

  def set_interval(self, key, interval):
    summary = self.lazy_get_summary(key)
    summary.interval = interval

  def lazy_get_summary(self, key):
    if key not in self._summaries.keys():
      self._summaries[key] = MovingAverageSummary()
    summary = self._summaries[key]
    return summary

  def update(self, key, value, writer=None, batch_type='training', batch=None, prefix='global'):

    summary = self.lazy_get_summary(key)
    average = summary.update(value)

    # Optionally write summaries of these values
    if writer is not None:
      tag = tf_get_summary_tag(batch_type, prefix, key)
      tag_avg = tf_get_summary_tag(batch_type, prefix, key + '_mean')
      tf_write_scalar_summary(writer, tag, batch, value)
      tf_write_scalar_summary(writer, tag_avg, batch, average)

    return average

class MovingAverageSummary(object):
  """
  Manages the online calculation of one statistic and summaries.
  Helper for Workflows that need this feature.
  """

  def __init__(self):
    self.samples = []
    self.interval = 100

  def clear(self):
    self.samples = []

  def update(self, value):
    # Track per-batch accuracy & compute average accuracy ever N batches
    if value is None:
      return None
    
    mean = None
    if len(self.samples) == self.interval:
      mean = np.mean(self.samples)
      self.clear()

    self.samples.append(value)
    return mean
