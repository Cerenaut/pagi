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

"""DualData and Dual classes."""

import numpy as np
import tensorflow as tf


class DualData(object):
  """
  DualData is a concept that arises from our need to recurrently feed data into TensorFlow graphs, that can't be
  recurrent.
  As a result we end up needing to manage off-graph (NumPy) tensors, and on-graph (TensorFlow) tensors, and defaults
  / initialization and copying between them. This class is supposed to make that easier and more systematic.
  """

  def __init__(self, root_name):
    self._duals = {}
    self._root_name = root_name

  def set_op(self, name, op):
    if name in self._duals.keys():
      dual = self._duals[name]
    else:
      dual = self.add(name)
      dual.set_op(op)

  def get_pl(self, name):
    if name in self._duals.keys():
      dual = self._duals[name]
      return dual.get_pl()
    return None

  def get_op(self, name):
    if name in self._duals.keys():
      dual = self._duals[name]
      return dual.get_op()
    return None

  def get_values(self, name):
    if name in self._duals.keys():
      dual = self._duals[name]
      return dual.get_values()
    return None

  def get(self, name):
    if name in self._duals.keys():
      return self._duals[name]
    return None

  def add(self, name, shape=None, default_value=None):
    if name not in self._duals:
      dual = Dual(name, shape, default_value)
      self._duals[name] = dual
    return self._duals[name]

  def add_dict(self, feed_dict, names):
    num_names = len(names)
    for i in range(0, num_names):
      name = names[i]
      dual = self._duals[name]
      pl = dual.get_pl()
      values = dual.get_values()
      feed_dict.update({
        pl: values,
      })

  def add_fetches(self, fetches, names):
    """ warning: replaces existing fetches """
    leaf_fetches = self.get_fetches(names)
    fetches[self._root_name] = leaf_fetches

  def get_fetches(self, names):
    fetches = {}
    num_names = len(names)
    for i in range(0, num_names):
      name = names[i]
      dual = self._duals[name]
      op = dual.get_op()
      fetches[name] = op
    return fetches

  def set_fetches(self, fetched, names):
    leaf_fetched = fetched[self._root_name]
    num_names = len(names)
    for i in range(0, num_names):
      name = names[i]
      dual = self._duals[name]
      values = leaf_fetched[name]
      dual.set_values(values)


class Dual(object):
  """
  This is a single tensor concept, that needs to exist in 2 places (TF graph, and off-graph).
  """

  def __init__(self, name, shape=None, default_value=None):
    self._name = name
    self._shape = shape
    self._op = None
    self._pl = None
    self._values = None
    if default_value is not None:
      self.set_values_to(default_value)

  def get_shape(self):

    # if shape is not defined, try to figure it out
    if self._shape is None:
      if self._op is not None:
        self._shape = self._op.get_shape()
      elif self._pl is not None:
        self._shape = self._pl.get_shape()

    return self._shape

  def add_pl(self, shape=None, name=None, default=False, dtype=tf.float32):
    """Create a new placeholder for this Dual if not already defined in graph."""
    if self._pl is None:
      if name is None:
        name = self._name + '_pl'
      if shape is None:
        shape = self._shape
      if default:
        self._pl = tf.placeholder_with_default(tf.cast(self._values, dtype), shape=shape, name=name)
      else:
        self._pl = tf.placeholder(dtype=dtype, shape=shape, name=name)
    return self._pl

  def get_pl(self):
    return self._pl

  def set_op(self, op):
    self._op = op

  def get_op(self):
    return self._op

  def get_values(self):
    return self._values

  def set_values(self, values):
    self._values = values.copy()

  def set_values_to(self, value):
    if isinstance(value, str):
      values = value
    else:
      shape = self.get_shape()
      values = np.zeros(shape)
      values.fill(value)
    self._values = values

  def set_values_uniform_rand(self, offset=0.0, scale=1.0):
    shape = self.get_shape()
    r = np.random.random_sample(shape)
    r = r + offset
    r = r * scale
    self._values = r
