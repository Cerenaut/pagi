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

"""
Helper methods for manipulating images.
Some methods are adapted from https://github.com/hmishra2250/NTM-One-Shot-TF
"""


import os
import io
import math
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize
from scipy.ndimage import rotate, shift
from skimage.draw import line_aa


def degrade_image(image, label=None, degrade_type='horizontal', degrade_value=0, degrade_factor=0.5, random_value=0.0):
  """Degrade the image by randomly removing one of image halves."""
  image_shape = image.shape.as_list()
  image_size = np.prod(image_shape[1:])

  if degrade_type == 'vertical' or degrade_type == 'horizontal':

    # # Miconi method
    # image = tf.reshape(image, [-1, image_size])
    #
    # preserved = np.ones(image_size)
    # preserved[int(image_size / 2):] = 0                  # [ 1's, 0's ]
    #
    # degraded_vals = np.ones(image_size)
    # degraded_vals[:] = degrade_value
    # degraded_vals[:int(image_size/2)] = 0               # [ 0's, dv's ]   # dv = degrade value

    # deal with 1d samples (or 2d image)
    if len(image_shape) == 2:
      preserved = np.ones(image_size)
      degraded_vals = np.zeros(image_size)
      degraded_vals[:] = degrade_value

      preserved[int(image_size / 2):] = 0
      degraded_vals[:int(image_size / 2)] = 0

    # 2d image
    else:
      sample_shape = image_shape[1:]
      width = image_shape[1]
      height = image_shape[2]

      preserved = np.ones(sample_shape)
      degraded_vals = np.zeros(sample_shape)
      degraded_vals[:] = degrade_value

      if degrade_type == 'vertical':
        # the whole row (width), half the columns (height)
        preserved[:, int(width/2):] = 0
        degraded_vals[:, 0:int(width/2)] = 0

      if degrade_type == 'horizontal':
        # half the row (width), all the columns (width)
        preserved[int(height/2):, :] = 0
        degraded_vals[0:int(height/2), :] = 0

    preserved = tf.convert_to_tensor(preserved, dtype=image.dtype)
    degraded_vals = tf.convert_to_tensor(degraded_vals, dtype=image.dtype)

    # Use random number generator, or specified random value
    rand_value = tf.cond(tf.cast(random_value, tf.float32) > 0,
                         lambda: random_value,
                         lambda: tf.random_uniform([]))

    # Randomly remove either half of the image
    rand_half = rand_value < .5  # randomly choose a half
    preserved = tf.cond(rand_half, lambda: 1 - preserved, lambda: preserved)  # swap 1's and 0's
    degraded_vals = tf.cond(rand_half, lambda: degrade_value - degraded_vals,
                            lambda: degraded_vals)  # swap dv's and 0's
    degraded_image = image * preserved  # zero out non-preserved bits
    degraded_image = degraded_image + degraded_vals  # add degrade_value at appropriate places (where it was zero)

    degraded_image = tf.reshape(degraded_image, image_shape, name='degraded_image')

  else:  # random
    preserve_mask = np.ones(image_size)
    preserve_mask[:int(degrade_factor * image_size)] = 0
    preserve_mask = tf.convert_to_tensor(preserve_mask, dtype=tf.float32)
    preserve_mask = tf.random_shuffle(preserve_mask)

    degrade_vec = np.ones(image_size)
    degrade_vec[:] = degrade_value
    degrade_vec = tf.convert_to_tensor(degrade_vec, dtype=tf.float32)
    degrade_vec = degrade_vec * preserve_mask  # preserved bits = degrade_value
    degrade_vec = degrade_value - degrade_vec  # preserved bits = 0, else degraded_value (flipped)

    image = tf.reshape(image, [-1, image_size])
    degraded_image = tf.multiply(image, preserve_mask)  # use broadcast to element-wise multiply batch with 'preserved'
    degraded_image = degraded_image + degrade_vec  # set non-preserved values to the 'degrade_value'
    degraded_image = tf.reshape(degraded_image, image_shape, name='degraded_image')

  if label is None:
    return degraded_image
  return degraded_image, label


def add_image_noise_flat(image, label=None, minval=0., noise_type='sp_binary', noise_factor=0.2):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_imagie_noise()"""
  image_shape = image.shape.as_list()
  image = tf.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_noise(image, label, minval, noise_type, noise_factor)
  image = tf.reshape(image, (-1, image_shape[1]))
  return image


def add_image_noise(image, label=None, minval=0., noise_type='sp_binary', noise_factor=0.2):
  image_shape = image.shape.as_list()
  image_size = np.prod(image_shape[1:])

  if noise_type == 'sp_float' or noise_type == 'sp_binary':
    noise_mask = np.zeros(image_size)
    noise_mask[:int(noise_factor * image_size)] = 1
    noise_mask = tf.convert_to_tensor(noise_mask, dtype=tf.float32)
    noise_mask = tf.random_shuffle(noise_mask)
    noise_mask = tf.reshape(noise_mask, [-1, image_shape[1], image_shape[2], image_shape[3]])

    noise_image = tf.random_uniform(image_shape, minval, 1.0)
    if noise_type == 'sp_binary':
      noise_image = tf.sign(noise_image)
    noise_image = tf.multiply(noise_image, noise_mask)  # retain noise in positions of noise mask

    image = tf.multiply(image, (1-noise_mask))  # zero out noise positions
    corrupted_image = image + noise_image       # add in the noise
  else:
    if noise_type == 'none':
      raise RuntimeWarning("Add noise has been called despite noise_type of 'none'.")
    else:
      raise NotImplementedError("The noise_type '{0}' is not supported.".format(noise_type))

  if label is None:
    return corrupted_image

  return corrupted_image, label


def pad_image(image, padding, mode='constant'):
  dim_pad = [padding, padding]
  paddings = tf.constant([dim_pad, dim_pad, [0, 0]])  # Avoid padding image channel
  return tf.pad(image, paddings, mode)


def shift_image(image, shift):
  shifts = []
  for i in np.arange(-shift, shift + 1):
    for j in np.arange(-shift, shift + 1):
      shifts.append([i, j])

  # Get random shift from list of potential shifts
  shifts = tf.convert_to_tensor(shifts, dtype=image.dtype)
  shuffled_shifts = tf.random_shuffle(shifts)
  random_shift = shuffled_shifts[0]

  return tf.contrib.image.translate(image, random_shift)


def get_shuffled_images(paths, labels, nb_samples=None):
  if nb_samples is not None:
    sampler = lambda x: random.sample(x, nb_samples)
  else:
    sampler = lambda x: x

  images = [(i, os.path.join(path, image)) for i, path in zip(labels, paths) for image in sampler(os.listdir(path))]
  random.shuffle(images)
  return images


def time_offset_label(labels_and_images):
  labels, images = zip(*labels_and_images)
  time_offset_labels = (None,) + labels[:-1]
  return zip(images, time_offset_labels)


def load_transform(image_path, angle=0., s=(0, 0), size=(20, 20)):
  """Transforms an image by rotating, shifting, resizing and inverting."""
  # Load the image
  original = imread(image_path, flatten=True)
  # Rotate the image
  rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
  # Shift the image
  shifted = shift(rotated, shift=s)
  # Resize the image
  resized = np.asarray(imresize(rotated, size=size),
                       dtype=np.float32) / 255  # Note here we coded manually as np.float32, it should be tf.float32
  # Invert the image
  inverted = 1. - resized
  max_value = np.max(inverted)
  if max_value > 0:
    inverted /= max_value
  return inverted


def generate_image_labels(num=10, size_x=24, size_y=24):
  """Generate num labels in the form of a unique small image."""

  image_labels = []
  delta = int(np.floor(size_y/num))
  print("delta = {0}".format(delta))
  print("num*delta = {0}".format(num * delta))
  for y in range(0, num * delta, delta):
    img = np.zeros((size_y, size_x), dtype=np.double)
    yy = y + int(delta*0.5)
    rr, cc, val = line_aa(yy, 0, yy, size_x-1)
    img[rr, cc] = val
    image_labels.append(img)
  return image_labels


def square_image_shape_from_1d(filters):
  """
  Make 1d tensor as square as possible. If the length is a prime, the worst case, it will remain 1d.
  Assumes and retains first dimension as batches.
  """
  height = int(math.sqrt(filters))

  while height > 1:
    width_remainder = filters % height
    if width_remainder == 0:
      break
    else:
      height = height - 1

  width = filters // height
  area = height * width
  lost_pixels = filters - area

  shape = [-1, height, width, 1]
  return shape, lost_pixels


def make_image_summary_shape_from_2d_shape(shape):
  """
  If you have a 2d tensor of (width, height) that you want to view as batch of grayscale images, use this.
  return [-1 width, height, 1]
  """
  shape.insert(0, -1)  # as many batches as exist
  shape.append(1)  # channels = 1 (grayscale)
  return shape


def make_image_summary_shape_from_2d(tensor):
  """
  If you have a 2d tensor of (width, height) that you want to view as batch of grayscale images, use this.
  return [-1 width, height, 1]
  """
  shape = tensor.get_shape().as_list()
  shape = make_image_summary_shape_from_2d_shape(shape)
  return shape


def get_image_summary_shape(tensor_shape):
  """
  Convert tensor_shape into an image shape to be shown in summary.
  Assumes tensor is already suitable to be shown as batch of images, and ensures the 4'th dimension is 1.
  :param tensor_shape assumes shape [batch, dim1, dim2, dim3].
  :return shape with dimension [batch, dim1, dim2-3, 1]
  """
  from copy import deepcopy
  # Rules for image summary: "Tensor must be 4-D with last dim 1, 3, or 4" (so, basically 1 then)
  summary_shape = deepcopy(tensor_shape)

  width = tensor_shape[2]
  depth = tensor_shape[3]
  if depth > 1:
    width = width * depth
    depth = 1

  summary_shape[2] = width
  summary_shape[3] = depth
  return summary_shape


def add_square_as_square(summaries, tensor, name):
  """ Convenience function for adding a square image to a summary. """
  image_shape = make_image_summary_shape_from_2d(tensor)
  image = tf.reshape(tensor, image_shape)
  summaries.append(tf.summary.image(name, image))


def array_to_image_string(image_array):
  """
  Converts a NumPy array representing an image to an encoded image string to be used in tf.Summary.Image().
  """
  num_dims = len(image_array.shape)

  if num_dims != 3:
    raise ValueError('Expecting 3 dimensions (height, weight, channel). Found {0} dimensions.'.format(num_dims))

  cmap = None
  if image_array.shape[2] == 1:
    cmap = 'gray'

  image_array = np.squeeze(image_array, axis=2)

  output = io.BytesIO()
  plt.imsave(output, image_array, format='PNG', cmap=cmap)
  image_string = output.getvalue()
  output.close()

  return image_string


def arbitrary_image_summary(summary, input_tensor, name='image', max_outputs=3, image_names=None):
  """
  Creates an off-graph tf.Summary.Image using arbitrary inputs.

  input_tensor contains multiple images.
    max_outputs specifies how many to plot
    OR  specify how many to plot by specifying their names in `image_names`
    num_images gets preference if it is defined.
  """

  if image_names is not None:
    max_outputs = len(image_names)

  num_outputs = min(max_outputs, input_tensor.shape[0])
  for i in range(num_outputs):
    image_array = input_tensor[i]
    h, w, c = image_array.shape

    image_string = array_to_image_string(image_array)

    image = tf.Summary.Image(
        height=h,
        width=w,
        colorspace=c,
        encoded_image_string=image_string)

    if image_names is not None:
      image_name = image_names[i]
    else:
      image_name = str(i)

    summary.value.add(tag=name + '/' + image_name, image=image)

  return summary


def add_op_images(dual, op_names, shape, max_outputs, summaries):
  """
  Convenience method to add a list of ops (as images) to a summary.

  @:param shape list of shapes (same lengths as op_names, or if same shape for all, then a single value
  @:param summaries are mutated
  """

  if not isinstance(shape, list):
    op_shapes = [shape] * len(op_names)
  else:
    op_shapes = shape

  for op_name, op_shape in zip(op_names, op_shapes):
    op = dual.get_op(op_name)
    if op is not None:
      reshaped = tf.reshape(op, shape)
      summaries.append(tf.summary.image(op_name, reshaped, max_outputs=max_outputs))


def add_arbitrary_images_summary(summary, scope_name, images, names, combined=False, max_outputs=3):
  """Add multiple images to a summary off graph, optionally combine into one."""
  if not combined:
    for image, name in zip(images, names):
      arbitrary_image_summary(summary, image, name='pcw/' + name, max_outputs=max_outputs)
  else:
    combined_image = None
    combined_name = ''
    for image, name in zip(images, names):
      if combined_image is None:
        combined_image = image
        combined_name = name
        continue
      combined_image = np.concatenate((combined_image, image), axis=1)
      combined_name = combined_name + '-' + name
    arbitrary_image_summary(summary, combined_image, name=scope_name + '/' + combined_name, max_outputs=max_outputs)


def concat_images(images, batch_size, images_shape=None):
  """
  Concatenate a list of images into one column of sub-images.
  Adds a 1 pixel line delineating them.

  If images_shape is not specified, use the shape of the first image

  :param images: a list of images
  :param batch_size:
  :param images_shape: first dimension is ignored (it is often not valid during graph build time)
  :return: the image containing concatenation of the images in `images`
  """
  concat_image = None

  if len(images) == 0:
    return

  if images_shape is None:
    images_shape = get_image_summary_shape(images[0].get_shape().as_list())

  for im in images:
    image_reshaped = tf.reshape(im, images_shape)
    if concat_image is None:
      concat_image = image_reshaped
    else:
      # add a line in between
      line = tf.ones([batch_size, 1, images_shape[2], images_shape[3]])
      concat_image = tf.concat([concat_image, line], axis=1)
      concat_image = tf.concat([concat_image, image_reshaped], axis=1)

  return concat_image
