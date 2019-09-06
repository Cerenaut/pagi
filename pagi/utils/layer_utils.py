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

import numpy as np
import tensorflow as tf


def type_activation_fn(fn_type):
  """Simple switcher for choosing activation functions."""
  if fn_type == 'none':
    fn = None
  elif fn_type == 'relu':
    fn = tf.nn.relu
  elif fn_type in ['leaky-relu', 'leaky_relu']:
    fn = tf.nn.leaky_relu
  elif fn_type == 'tanh':
    fn = tf.tanh
  elif fn_type == 'sigmoid':
    fn = tf.sigmoid
  elif fn_type == 'softmax':
    fn = tf.nn.softmax
  elif fn_type == 'logistic':
    fn = tf.logistic
  else:
    raise NotImplementedError(
        'Activation function implemented: ' + str(fn_type))

  return fn


def activation_fn(x, fn_type):
  """Activation function switcher."""
  fn = type_activation_fn(fn_type)
  output = fn(x) if fn is not None else x

  return output, fn


def max_pool_with_argmax(inputs, pool_size, stride, padding='SAME'):
  """
  The default `tf.nn.max_pool_with_argmax` does does not provide gradient
  operation. This method utilises `tf.nn.max_pool_with_argmax` to extract
  pooling indices, and applying max pooling using`tf.layers.max_pooling2d`
  instead.
  Reference: https://github.com/tensorflow/tensorflow/issues/2169
  """
  with tf.name_scope('MaxPoolArgmax'):
    _, mask = tf.nn.max_pool_with_argmax(
        inputs,
        ksize=[1, pool_size, pool_size, 1],
        strides=[1, stride, stride, 1],
        padding=padding)

    mask = tf.stop_gradient(mask, name='pooling_indices')
    outputs = tf.layers.max_pooling2d(inputs, pool_size, stride, padding)

    return outputs, mask


def pool(tensor, pool_size, pool_strides, unpooling_mode='fixed', padding='SAME'):
  """Perform the Max Pooling operation on a tensor."""
  mask = None

  if unpooling_mode == 'argmax':
    pooled, mask = max_pool_with_argmax(tensor, pool_size, pool_strides, padding=padding)
  else:
    pooled = tf.layers.max_pooling2d(tensor, pool_size, pool_strides, padding=padding)

  return pooled, mask


def unpool_with_argmax(inputs, indices, ksize, scope='unpool'):
  """
  Reverses the effects of the max pooling operation by placing the max values
  back into their original position, based on the indices retrieved from the
  `tf.max_pool_with_argmax` operation.
  Reference: https://github.com/tensorflow/tensorflow/issues/2169
  Args:
    inputs: A 4-D `Tensor` of shape [batch_size, width, height, num_channels]
      that represents the results of the pooling operation.
    indices: A 4D `Tensor` that represents the pooling indices from the pooling
      operation applied to `inputs`.
    ksize:  A list of `ints` that has length `>= 4`. The size of the window
      used in the max pooling operation.
    scope: A `string` to customise the name scope of the unpooling operation.
      Defaults to `unpool`.
  """
  with tf.variable_scope(scope):
    input_shape = tf.shape(inputs)
    output_shape = [input_shape[0],
                    input_shape[1] * ksize[1],
                    input_shape[2] * ksize[2],
                    input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(inputs, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64),
                                      dtype=indices.dtype),
                             shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(indices) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(indices, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = inputs.get_shape()
    set_output_shape = [set_input_shape[0],
                        set_input_shape[1] * ksize[1],
                        set_input_shape[2] * ksize[2],
                        set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret


def unpool_with_fixed(inputs, stride, mode='zeros', scope='unpool'):
  """
  Reverses the effects of the max pooling operation by placing the max values
  from the max pooling operation into a fixed position.
  Examples:
    [0.6] =>[[.6, 0],   (stride=2, mode='zeros')
            [ 0, 0]]
    [0.6] =>[[.6, .6],  (stride=2, mode='copy')
            [ .6, .6]]
  Reference: https://github.com/tensorflow/tensorflow/issues/2169
  Args:
    inputs: A 4-D `Tensor` of shape [batch_size, width, height, num_channels]
      that represents the results of the pooling operation.
    stride: An integer, specifying the strides of the pooling operation.
    mode: A string, either 'zeros' or 'copy', indicating which value to use
      for undefined cells. Case-insensitive.
    scope: A `string` to customise the name scope of the unpooling operation.
      Defaults to `unpool`.
  """
  modes = ['copy', 'zeros']
  mode = mode.lower()

  if isinstance(inputs, np.ndarray):
    inputs = tf.convert_to_tensor(inputs)

  def upsample_along_axis(volume, axis, stride, mode):
    shape = volume.get_shape().as_list()

    assert mode in modes
    assert 0 <= axis < len(shape)

    target_shape = shape[:]
    target_shape[axis] *= stride

    padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'zeros' else volume

    parts = [volume] + [padding for _ in range(stride - 1)]
    volume = tf.concat(parts, min(axis + 1, len(shape) - 1))
    volume = tf.reshape(volume, target_shape)

    return volume

  with tf.name_scope(scope):
    inputs = upsample_along_axis(inputs, 2, stride, mode=mode)
    inputs = upsample_along_axis(inputs, 1, stride, mode=mode)
    return inputs


def unpool(pooled_tensor, pool_size, pool_strides, indices=None, unpooling_mode='fixed',
           crop_unpooled=True, pre_pooled_shape=None):
  """
  Performs unpooling on an existing pooled tensor.

  Note: unpool_with_argmax uses indices/scatter_nd, and currently
  does not work with batch_size > 1. Keeping it here as it might come in
  handy later if we ever need to restore the true max positions.
  """
  if unpooling_mode == 'argmax':
    logging.warning('Unpool with argmax is currently not working. See comments in code.')

    unpooled = unpool_with_argmax(pooled_tensor, indices, ksize=[1, pool_size, pool_size, 1])

  elif unpooling_mode == 'fixed':
    unpooled = unpool_with_fixed(pooled_tensor, pool_strides, mode='zeros')
  else:
    raise NotImplementedError('Unpooling method not supported: ' + str(unpooling_mode))

  # crop the unpooled value to match dimensions with pre-pooled version
  if crop_unpooled and pre_pooled_shape is not None:

    height = pre_pooled_shape[1]
    width = pre_pooled_shape[2]

    unpooled = tf.image.crop_to_bounding_box(unpooled, 0, 0, height, width)

    unpooled_shape = unpooled.shape.as_list()
    up_height = unpooled_shape[1]
    up_width = unpooled_shape[2]

    if up_height > height or up_width > width:
      logging.warning('Un-Pooled volume was cropped, up-height %d, up-width: %d, height %d, width: %d.',
                      up_height, up_width, height, width)

  return unpooled
