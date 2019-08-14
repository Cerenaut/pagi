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

"""Layer-related methods."""

import tensorflow as tf


def type_activation_fn(fn_type):
  """Simple switcher for choosing activation functions."""
  if fn_type == 'none':
    fn = None
  elif fn_type == 'relu':
    fn = tf.nn.relu
  elif fn_type == 'leaky-relu':
    fn = tf.nn.leaky_relu
  elif fn_type == 'tanh':
    fn = tf.tanh
  elif fn_type == 'sigmoid':
    fn = tf.sigmoid
  elif fn_type == 'softmax':
    fn = tf.nn.softmax
  elif fn_type == 'logistic':
    fn = tf.logistic
  elif fn_type == 'leaky_relu':
    fn = tf.nn.leaky_relu
  else:
    raise NotImplementedError(
        'Activation function implemented: ' + str(fn_type))

  return fn

def activation_fn(x, fn_type):
  """Activation function switcher."""
  fn = type_activation_fn(fn_type)
  output = fn(x) if fn is not None else x

  return output, fn
