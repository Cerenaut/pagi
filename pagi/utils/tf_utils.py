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

"""Tensorflow-dependent helper methods."""

import logging

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils


def tf_invert(feature, label):
  inverted = 1.0 - feature
  return inverted, label


def tf_set_min(source, label, tgt_min, current_min=0):
  """
  For an input tensor with `current_min` as the minimum value, set a new minimum.
  **WARNING**: tgt_min CANNOT be zero, or it will not work
  """

  source = tf_print(source, message="source: ", summarize=280, mute=True)

  source_zeros = tf.to_float(tf.equal(source, current_min))   # 1.0(True) where =curr_min, 0.(False) where !=curr_min
  minvals_inplace = tgt_min * source_zeros
  target = source + minvals_inplace

  target = tf_print(target, message="target: ", summarize=280, mute=True)

  return target, label


def tf_label_filter(feature, label, label_list):  # pylint: disable=W0613
  keep_labels = tf.constant(label_list)
  tile_multiples = tf.concat([
      tf.ones(tf.shape(tf.shape(label)), dtype=tf.int32),
      tf.shape(keep_labels)], axis=0)
  label_tile = tf.tile(tf.expand_dims(label, -1), tile_multiples)
  return tf.reduce_any(tf.equal(label_tile, keep_labels), -1)


def tf_get_sub_area(tensor, start_dim=1):
  shape = tensor.get_shape().as_list()
  area = np.prod(shape[start_dim:])
  return area


def tf_get_area(tensor):
  shape = tensor.get_shape().as_list()
  area = np.prod(shape[0:])
  return area


def tf_dog_kernel(size, mid, std, k):
  """
  if k is not defined, use the convention for the Laplacian of Guassians, which is 1.6
  https://en.wikipedia.org/wiki/Difference_of_Gaussians
  """

  g1 = tf_gaussian_kernel(size, mid, std)
  g2 = tf_gaussian_kernel(size, mid, k * std)
  dog = g1 - g2
  return dog


def tf_gaussian_kernel(size: int, mean: float, std: float):
  """Makes 2D gaussian Kernel for convolution."""
  # https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
  d = tf.distributions.Normal(mean, std)
  vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
  gauss_kernel = tf.einsum('i,j->ij', vals, vals)
  return gauss_kernel / tf.reduce_sum(gauss_kernel)


def tf_build_top_k_mask_op(input_tensor, k, batch_size, input_area):

  # Find the "winners". The top k elements in each batch sample. this is
  # what top_k does.
  # ---------------------------------------------------------------------
  input_vector = tf.reshape(input_tensor, [batch_size, input_area])
  top_values, top_indices = tf.nn.top_k(input_vector, k=k)  

  # Now the problem is that sparse_to_dense requires a 1d vector, or a
  # vector of n-d indices. So we will need to give it a 1d vector, but
  # offset the indices.
  # Since we have k "winners" per batch item. All k need to be offset by
  # b * cells.
  # ---------------------------------------------------------------------
  indices_offsets = np.empty([batch_size * k], dtype=int)
  for b in range(batch_size):  # foreach( batch of k winners )
    for n in range(k):  # foreach( winner in the batch )
      index = b * k + n  # each b is a vector of k indices
      offset = b * input_area  # when we offset
      indices_offsets[index] = offset
  indices_offsets = tf.convert_to_tensor(indices_offsets, dtype=tf.int32)  # convert the vector to a TF vector

  # Now that we have the batch x indices vector, make it a 1d vector.
  # Then add the offsets
  # ---------------------------------------------------------------------
  indices_vector = tf.reshape(top_indices, [batch_size * k])
  indices_vector = tf.add(indices_vector, indices_offsets)

  # Finally, construct the mask. We need a default vector the the output
  # which is 1s for each filled element.
  # ---------------------------------------------------------------------
  values_vector = tf.ones(batch_size * k, dtype=tf.float32)
  mask_vector_dense = tf.sparse_to_dense(indices_vector, [batch_size * input_area], values_vector, default_value=0, validate_indices=False)
  batch_mask_vector_op = tf.reshape(mask_vector_dense, [batch_size, input_area])  #, name="rank-mask")

  return batch_mask_vector_op


def tf_build_top_k_mask_4d_op(input_tensor, k, batch_size, h, w, input_area):

  batch_h_w_size = batch_size * h * w

  logging.debug('encoding shape = (%s, %s, %s, %s)', batch_size, h, w, input_area)

  # Find the "winners". The top k elements in each batch sample. this is
  # what top_k does.
  # ---------------------------------------------------------------------
  input_vector = tf.reshape(input_tensor, [batch_h_w_size, input_area])
  top_values, top_indices = tf.nn.top_k(input_vector, k=k) # top_k per input_area

  # Now the problem is that sparse_to_dense requires a 1d vector, or a
  # vector of n-d indices. So we will need to give it a 1d vector, but
  # offset the indices.
  # Since we have k "winners" per batch item. All k need to be offset by
  # b * cells.
  # ---------------------------------------------------------------------
  indices_offsets = np.empty([batch_h_w_size * k], dtype=int)
  for b in range(batch_size):  # foreach( batch of k winners )
    for y in range(h): 
      for x in range(w): 
        for n in range(k):  # foreach( winner in the batch )
          #index = b * k + n  # each b is a vector of k indices
          #offset = b * input_area  # when we offset

          index = b * k * w * h \
                + y * k * w     \
                + x * k         \
                + n

          offset = b * input_area * w * h \
                 + y * input_area * w     \
                 + x * input_area

          indices_offsets[index] = offset

  indices_offsets = tf.convert_to_tensor(indices_offsets, dtype=tf.int32)  # convert the vector to a TF vector

  # Now that we have the batch x indices vector, make it a 1d vector.
  # Then add the offsets
  # ---------------------------------------------------------------------
  indices_vector = tf.reshape(top_indices, [batch_h_w_size * k])
  indices_vector = tf.add(indices_vector, indices_offsets)

  # Finally, construct the mask. We need a default vector the the output
  # which is 1s for each filled element.
  # ---------------------------------------------------------------------
  values_vector = tf.ones(batch_h_w_size * k, dtype=tf.float32)
  mask_vector_dense = tf.sparse_to_dense(indices_vector, [batch_h_w_size * input_area], values_vector, default_value=0, validate_indices=False)
  batch_mask_vector_op = tf.reshape(mask_vector_dense, [batch_size, h, w, input_area]) #, name="rank-mask")

  return batch_mask_vector_op


def tf_reduce_var(x, axis=None, keepdims=False):
  """Variance of a tensor, alongside the specified axis.
  Stolen from: https://stackoverflow.com/a/43409235
  # Arguments
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.
  # Returns
      A tensor with the variance of elements of `x`.
  """
  m = tf.reduce_mean(x, axis=axis, keepdims=True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def tf_build_stats_summaries(tensor, name_scope):
  """Build statistical summaries for a specific variable/tensor."""
  with tf.name_scope(name_scope):
    m_mean = tf.reduce_mean(tensor)
    m_var = tf_reduce_var(tensor)
    m_min = tf.reduce_min(tensor)
    m_max = tf.reduce_max(tensor)
    m_sum = tf.reduce_sum(tensor)

    mean_op = tf.summary.scalar('mean', m_mean)
    sd_op = tf.summary.scalar('sd', tf.sqrt(m_var))
    min_op = tf.summary.scalar('min', m_min)
    max_op = tf.summary.scalar('max', m_max)
    sum_op = tf.summary.scalar('sum', m_sum)

    stats_summaries = []
    stats_summaries.append(mean_op)
    stats_summaries.append(sd_op)
    stats_summaries.append(min_op)
    stats_summaries.append(max_op)
    stats_summaries.append(sum_op)

    return stats_summaries


def tf_build_stats_summaries_short(tensor, name_scope):
  """
  Build a shorter version of statistical summaries for a specific variable/tensor.
  Mean, StdDev, Min and Max
  """

  with tf.name_scope(name_scope):
    m_mean = tf.reduce_mean(tensor)
    m_var = tf_reduce_var(tensor)
    m_min = tf.reduce_min(tensor)
    m_max = tf.reduce_max(tensor)

    mean_op = tf.summary.scalar('mean', m_mean)
    sd_op = tf.summary.scalar('sd', tf.sqrt(m_var))
    min_op = tf.summary.scalar('min', m_min)
    max_op = tf.summary.scalar('max', m_max)

    stats_summaries = []
    stats_summaries.append(mean_op)
    stats_summaries.append(sd_op)
    stats_summaries.append(min_op)
    stats_summaries.append(max_op)

    return stats_summaries


def tf_summary_scalar(name="name", tensor=None, mute=True):
  """Convenience method for creating scalar summary with mute option."""
  if not mute:
    tf.summary.scalar(name=name, tensor=tensor)
  else:
    pass


def tf_print(var, message="", summarize=10, mute=True):
  """Convenience function for printing tensors in graph at runtime, also better formatting, can be muted."""
  if not mute:
    message = "\n" + message + "\n\t"
    return tf.Print(var, [var], message=message, summarize=summarize)
  else:
    return var


def degrade_by_mask_per_bit(input_tensor, degrade_mask=None, degrade_factor=0.5, degrade_value=0.0, label=None):
  """
  Randomly degrade 'degrade_factor' proportion of bits of the `degrade_mask`, resetting them to 'reset_value'.
  *** Currently - It does the exact same degradation pattern for every input in the batch.***
  First dimension of input must be batch.
  """

  input_shape = input_tensor.shape.as_list()
  input_size = np.prod(input_shape[1:])
  input_tensor = tf.reshape(input_tensor, [-1, input_size])

  # generate random values between 0 and 1, for all positions in the mask, then use it to select approp. proportion
  random_values = tf.random_uniform(shape=[input_size])
  if degrade_mask is not None:
    random_values = random_values * degrade_mask
  preserve_mask = tf.greater(random_values, degrade_factor)
  preserve_mask = tf.to_float(preserve_mask)

  degrade_vec = degrade_value * preserve_mask
  degrade_vec = degrade_value - degrade_vec        # preserved bits = 0, else degraded_value (flipped)

  degraded = tf.multiply(input_tensor, preserve_mask)  # use broadcast to element-wise multiply batch with 'preserved'
  degraded = degraded + degrade_vec  # set non-preserved values to the 'degrade_value'
  degraded = tf.reshape(degraded, input_shape)

  if label is None:
    return degraded
  return degraded, label


def degrade_by_mask(input_tensor, num_active, degrade_mask=None, degrade_factor=0.5, degrade_value=0.0, label=None):
  """

  WARNING - this version works with mask, but only works if the min value is 0 (such as -1)
            No point updating it now.

  Randomly degrade degrade_factor bits, resetting them to 'reset_value'.
  *** Currently - It does the exact same degradation pattern for every input in the batch.***
  First dimension of input must be batch.
  """

  dbug = False

  if dbug:
    num_active = int(num_active)
    print("num_active = " + str(num_active))

  input_shape = input_tensor.shape.as_list()
  input_size = np.prod(input_shape[1:])
  input_tensor = tf.reshape(input_tensor, [-1, input_size])

  input_tensor = tf_print(input_tensor, "input_tensor", mute=not dbug)

  # make a compact version of the active bits, and randomly knockout half the bits
  preserved_compact = np.ones(num_active)
  preserved_compact[:int(degrade_factor * num_active)] = 0
  preserved_compact = tf.convert_to_tensor(preserved_compact, dtype=tf.float32)
  preserved_compact = tf.random_shuffle(preserved_compact)

  preserved_compact = tf_print(preserved_compact, "preserved_compact", mute=not dbug)

  # map back to the actual positions, use to clear all not preserved (i.e. to degrade)
  _, indices_of_active = tf.nn.top_k(input=degrade_mask, k=num_active, sorted=False)
  indices_of_active = tf_print(indices_of_active, "indices_of_active", mute=not dbug)
  preserve_mask = tf.sparse_to_dense(sparse_indices=indices_of_active,
                                     output_shape=[input_size],
                                     sparse_values=preserved_compact,
                                     default_value=0,
                                     validate_indices=False)

  preserve_mask = tf_print(preserve_mask, "preserve_mask", mute=not dbug)

  degrade_value_vec = np.ones(input_size)
  degrade_value_vec[:] = degrade_value
  degrade_value_vec = tf.convert_to_tensor(degrade_value_vec, dtype=tf.float32)
  degrade_value_vec = degrade_value_vec * preserve_mask        # preserved bits = degrade_value
  degrade_value_vec = degrade_value - degrade_value_vec        # preserved bits = 0, else degraded_value (flipped)

  degrade_value_vec = tf_print(degrade_value_vec, "degrade_value_vec", mute=not dbug)

  degraded = tf.multiply(input_tensor, preserve_mask)  # use broadcast to element-wise multiply batch with 'preserved'
  degraded = degraded + degrade_value_vec  # set non-preserved values to the 'degrade_value'

  degraded = tf_print(degraded, "degraded", mute=not dbug)

  degraded = tf.reshape(degraded, input_shape)

  if label is None:
    return degraded
  return degraded, label


# from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
def histogram_summary(tag, values, bins=1000, minimum=None, maximum=None):
  """Logs the histogram of a list/vector of values."""

  # Convert to a numpy array
  values = np.array(values)

  if minimum is None:
    minimum = float(np.min(values))

  if maximum is None:
    maximum = float(np.max(values))

  # Create histogram using numpy
  counts, bin_edges = np.histogram(values, bins=bins, range=[minimum, maximum])

  # Fill fields of histogram proto
  hist = tf.HistogramProto()
  hist.min = minimum
  hist.max = maximum
  hist.num = int(np.prod(values.shape))
  hist.sum = float(np.sum(values))
  hist.sum_squares = float(np.sum(values ** 2))

  # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
  # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
  # Thus, we drop the start of the first bin
  bin_edges = bin_edges[1:]

  # Add bin edges and counts
  for edge in bin_edges:
    hist.bucket_limit.append(edge)
  for c in counts:
    hist.bucket.append(c)

  # Create and write Summary
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])

  return summary

