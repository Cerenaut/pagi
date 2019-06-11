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

"""Numpy-dependent helper methods."""

import logging
import math
import os

import numpy as np

import matplotlib.pyplot as plt


def np_roulette(vector, selections, exclude=-1):
  """
  Aka fitness-proportional selection
  """
  total = np.sum(vector)
  elements = len(vector)
  selected = np.zeros(selections)

  for i in range(0, selections):
    r = np.random.rand() * total # reset target
    x_sum = 0.0 # reset sum
    for j in range(0, elements):
      x = vector[j]
      x_sum += x
      if x_sum >= r:
        if exclude >= 0:
          if j == exclude:
            continue
        selected[i] = j
        break

  return selected


def np_noise_salt_and_pepper(tensor, rate=0.3):
  """
  Salt and pepper noise.
  Can't be both.
  Use one rand() per pixel.
  So if rate=0.1, there should be a 0.5 chance of each
  i.e. if r < (rate/2) then salt, (1)
       if r > 1-(rate/2) then pepper (0)
  """
  half_rate = rate * 0.5
  inv_half_rate = 1.0 - half_rate
  r = np.random.uniform(0, 1, tensor.shape) # generate a random matrix
  pepper = np.where(r < half_rate)
  salt = np.where(r >= inv_half_rate)
  tensor[pepper] = 0.0
  tensor[salt] = 1.0
  return tensor


def np_dropout(mask, rate=0.1):
  """
  Remove bits from mask with probability rate
  """
  r = np.random.uniform(0, 1, mask.shape)  # generate a random matrix
  exclusions = np.where(r < rate)
  mask[exclusions] = 0.0
  return mask


def np_write_array_list_as_image(base_folder, arr, name):

  filetype = 'png'
  rel_filepath = name + '.' + filetype
  filepath = os.path.join(base_folder, rel_filepath)

  arr = np.squeeze(arr, 0)
  arr = np.squeeze(arr, 2)

  plt.imshow(arr, interpolation='none')
  plt.savefig(filepath, dpi=300, format=filetype)
  plt.close()


def np_write_filters(filters, filter_shape, file_name='filters.png'):
  """
  Uses Matplotlib to render all filters into a 2d array of tile images.
  Writes to disk.
  """

  plt.switch_backend('agg')
  plt.ioff()

  logging.debug('save filters, shape: %s', filters.shape)

  n_filters = filters.shape[0]
  n_columns = int(math.sqrt(n_filters))
  n_rows = int(n_filters / n_columns)

  if (n_rows * n_columns) < n_filters:
    n_rows += 1

  filter_h = filter_shape[0] + 1     # leave a border around filters, adding one pixel per filter
  filter_w = filter_shape[1] + 1
  grid_w = n_columns * filter_w - 1  # remove px border around edge of image (the last filter in row or col)
  grid_h = n_rows * filter_h - 1

  grid_image = np.zeros([grid_h, grid_w])
  for i in range(n_rows):
    for j in range(n_columns):

      k = i * n_columns + j
      if k >= n_filters:
        continue

      yf = i * filter_h
      xf = j * filter_w

      k_filter = filters[k]
      filter_2d = np.reshape(k_filter, filter_shape)
      filter_2d_norm = filter_2d / filter_2d.max()  # normalize each 2d filter

      for y in range(filter_h - 1):     # leave a border around filters by stopping one short (it's an extra px high)
        for x in range(filter_w - 1):
          px = filter_2d_norm[y, x]
          grid_image[yf + y, xf + x] = px

  plt.Figure(figsize=(100, 100))
  plt.title('All Filters')
  plt.imshow(grid_image, interpolation='none')
  plt.savefig(file_name, dpi=900)
  plt.close()

  logging.debug('save filters OK')


def print_simple_stats(arr, name):
  min = np.min(arr)
  max = np.max(arr)
  print('{0} (min,max) = ({1},{2})'.format(name, min, max))

