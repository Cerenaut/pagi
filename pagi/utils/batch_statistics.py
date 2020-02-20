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

"""BatchStatistics class"""

import numpy as np

from pagi.utils.tf_utils import tf_get_summary_tag
from pagi.utils.tf_utils import tf_write_scalar_summary


class BatchStatistics:
  """
  Manages the online calculation of batch statistics.
  """

  def __init__(self):
    self._statistics = {}

  def reset(self, key=None):
    if key is None:
      for _, s in self._statistics.items():
        s.reset()
    else:
      s = self.get_statistic(key)
      s.reset()

  def lazy_get_statistic(self, key):
    if key not in self._statistics.keys():
      self._statistics[key] = BatchStatistic()
    return self.get_statistic(key)

  def get_statistic(self, key):
    s = self._statistics[key]
    return s

  def update(self, key, sum_values, count_values):
    s = self.lazy_get_statistic(key)
    s.update(sum_values, count_values)

  def get_mean(self, key):
    s = self.lazy_get_statistic(key)
    return s.get_mean()

class BatchStatistic:
  """
  Accumulates a statistic for which multiple samples are accumulated at each batch.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self.sum = 0.0
    self.count = 0
    self.list = []
    self.min = None
    self.max = None

  def update(self, sum_values, count_values):
    """Add new samples to the stat."""
    self.sum += sum_values
    self.count += count_values
    self.list.append(sum_values)

    if self.min is None:
      self.min = sum_values
    elif sum_values < self.min: 
      self.min = sum_values

    if self.max is None:
      self.max = sum_values
    elif sum_values > self.max: 
      self.max = sum_values

  def get_mean(self):
    if self.count == 0:
      return float('NaN')
    mean = self.sum / self.count
    return mean

