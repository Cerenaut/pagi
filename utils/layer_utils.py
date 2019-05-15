# Copyright (C) 2018 Project AGI
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


# def unpool_2d(pool,
#               ind,
#               stride=[1, 2, 2, 1],
#               scope='unpool_2d'):
#   """Adds a 2D unpooling op.
#   https://arxiv.org/abs/1505.04366
#   Unpooling layer after max_pool_with_argmax.
#        Args:
#            pool:        max pooled output tensor
#            ind:         argmax indices
#            stride:      stride is the same as for the pool
#        Return:
#            unpool:    unpooling tensor
#   """
#   with tf.variable_scope(scope):
#     input_shape = tf.shape(pool)
#     output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]
#
#     flat_input_size = tf.reduce_prod(input_shape)
#     flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
#
#     pool_ = tf.reshape(pool, [flat_input_size])
#     batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
#                              shape=[input_shape[0], 1, 1, 1])
#     b = tf.ones_like(ind) * batch_range
#     b1 = tf.reshape(b, [flat_input_size, 1])
#     ind_ = tf.reshape(ind, [flat_input_size, 1])
#     ind_ = tf.concat([b1, ind_], axis=-1)
#
#     ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
#     ret = tf.reshape(ret, output_shape)
#
#     set_input_shape = pool.get_shape()
#     set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
#     ret.set_shape(set_output_shape)
#     return ret


def unpool_2d(pool, ind, stride=None, scope='unpool_2d'):
  """
  Discussion: https://github.com/tensorflow/tensorflow/issues/2169
  PR: https://github.com/tensorflow/tensorflow/issues/2169
  Code: https://github.com/rayanelleuch/tensorflow/blob/b46d50583d8f4893f1b1d629d0ac9cb2cff580af/tensorflow/contrib/layers/python/layers/layers.py#L2291-L2327

  Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """
  if stride is None:
    stride = [1, 2, 2, 1]
  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                             shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], axis=-1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret


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
