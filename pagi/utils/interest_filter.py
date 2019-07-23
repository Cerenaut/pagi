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

"""InteresetFilter class."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

from pagi.utils import tf_utils, image_utils


class InterestFilter:
  """Interest Filter."""

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        batch_size=0,
        num_features=0,

        # scaling
        scale_range=[1.0, 0.5, 0.25],

        # Kernels
        # -----------------
        # pe = positional encoding namespace
        pe_size=0,
        pe_std=0.0,

        # nms = non-maxima suppression
        nms_size=0,
        nms_stride=0.0,

        # f = feature
        f_size=0,
        f_std=0.0,
        f_k=1.6,      # approximates Laplacian of Guassian

        max_outputs=3
    )

  def __init__(self):
    self._image_dic = {}
    self._hparams = None
    self.scalar_vals = []  # (name, tensor of shape=[])

  def set_image(self, name, val):
    self._image_dic[name] = val

  def get_image(self, name):
    return self._image_dic[name]

  @staticmethod
  def smooth_kernel(size, std):
    mid = 0.0    # 0.5 * (size-1)
    return tf_utils.tf_gaussian_kernel(size, mid, std)

  @staticmethod
  def dog_kernel(size, std, k):
    mid = 0.0    # 0.5 * (size-1)
    return tf_utils.tf_dog_kernel(size, mid, std, k)

  def build(self, image_tensor, conv_encodings, hparams):
    """Build the interest filter."""
    self._hparams = hparams

    self.set_image('image', image_tensor)

    # get conv output at interest points
    fixation_mask = self.interest_function(image_tensor)    # input_image dims
    masked_encodings = tf.multiply(conv_encodings, fixation_mask)
    self.set_image('masked_encodings', masked_encodings)

    # prepare smoothing kernel
    smooth_k = InterestFilter.smooth_kernel(hparams.pe_size, hparams.pe_std)
    positional_encodings = self._depthwise_conv_with_2d_filter(masked_encodings, smooth_k)
    self.set_image('positional_encodings', positional_encodings)

    return fixation_mask, positional_encodings

  @staticmethod
  def im_name_scale(im_name, scale):
    return im_name + '_' + str(scale)

  def interest_function(self, image_tensor):
    """Interest function."""
    image_shape = image_tensor.get_shape().as_list()
    image_shape_wh = [image_shape[1], image_shape[2]]

    # get kernels for later use
    dog_k = InterestFilter.dog_kernel(self._hparams.f_size, self._hparams.f_std, self._hparams.f_k)
    dog_k = dog_k[:, :, tf.newaxis, tf.newaxis]

    def interest_core(positive=True):

      def im_name(name):
        return name + '_' + ('pos' if positive else 'neg')

      dog_kernel = dog_k
      if not positive:
        dog_kernel = -dog_k

      # analysis at multiple scales
      conv_sum = None
      for scale in self._hparams.scale_range:

        # DoG kernel - edge and corner detection plus smoothing
        zoom_image_shape = np.multiply(scale, image_shape_wh)
        image = tf.image.resize_images(image_tensor, zoom_image_shape)
        image = tf.nn.conv2d(image, dog_kernel, strides=[1, 1, 1, 1], padding='SAME')
        image = tf.image.resize_images(image, [image_shape[1], image_shape[2]])
        self.set_image(self.im_name_scale(im_name('features'), scale), image)

        # non-maxima suppression (i.e. blend the peaks) at this scale
        image = self._non_maxima_suppression(image)
        self.set_image(self.im_name_scale(im_name('non_maxima'), scale), image)

        # accumulate features at this scale
        if conv_sum is None:
          conv_sum = image
        else:
          conv_sum = conv_sum + image

      self.set_image(im_name('conv_sum'), conv_sum)

      # sparse filtering
      k = self._hparams.num_features
      batch_size = self._hparams.batch_size
      input_area = image_shape[1] * image_shape[2]
      fixations_mask_1d = tf_utils.tf_build_top_k_mask_op(conv_sum, k, batch_size, input_area)
      fixations_mask = tf.reshape(fixations_mask_1d, image_shape)

      self.set_image(im_name('fixation_mask'), fixations_mask)

      return fixations_mask

    fixs_mask_pos = interest_core(positive=True)
    fixs_mask_neg = interest_core(positive=False)

    fixs_mask = fixs_mask_neg + fixs_mask_pos
    self.set_image('fixation_mask', fixs_mask)

    return fixs_mask

  def _non_maxima_suppression(self, image):
    """Non maxima supression."""
    use_smoothing = False
    if use_smoothing:
      smooth_k = InterestFilter.smooth_kernel(self._hparams.nms_size, self._hparams.nms_std)
      smooth_k = smooth_k[:, :, tf.newaxis, tf.newaxis]
      image = tf.nn.conv2d(image, smooth_k, strides=[1, 1, 1, 1], padding='SAME')
      return image

    pool_size = self._hparams.nms_size
    stride = self._hparams.nms_stride
    padding = 'SAME'

    strides = [1, stride, stride, 1]
    ksize = [1, pool_size, pool_size, 1]

    # image, indices = tf.nn.max_pool_with_argmax(image, ksize=ksize, strides=strides, padding=padding)
    #
    # self.scalar_vals.append(('indices_max', tf.reduce_max(indices)))
    # self.scalar_vals.append(('image_max', tf.reduce_max(image)))
    #
    # print('non maxima pooled image: ', image)
    # print('non maxima indices: ', indices)
    #
    # image = layer_utils.unpool_2d(image, indices, strides)
    # print('non maxima unpooled image: ', image)

    # The unpooling output is also the gradient of the pooling operation
    # So do pool and extract unpool from grads:
    # https://assiaben.github.io/posts/2018-06-tf-unpooling/
    img_op = image
    pool_op = tf.nn.max_pool(img_op, ksize=ksize, strides=strides, padding=padding, name='pool')
    unpool_op = gen_nn_ops.max_pool_grad(img_op, pool_op, pool_op, ksize, strides, padding)
    image = unpool_op

    return image

  def add_summaries(self, summaries):
    """Add TensorBoard summaries."""
    max_outputs = self._hparams.max_outputs
    show_all_separately = False

    def show_image(im_name, im):
      shape = image_utils.get_image_summary_shape(im.get_shape().as_list())
      reshaped = tf.reshape(im, shape)
      summaries.append(tf.summary.image(im_name, reshaped, max_outputs=max_outputs))

    def concat_images(concat_image_name, im_names, im_shape=None):
      if not im_names:
        return

      if im_shape is None:
        im = self._image_dic[im_names[0]]
        im_shape = image_utils.get_image_summary_shape(im.get_shape().as_list())

      images = []
      for im_name in im_names:
        im = self._image_dic[im_name]
        images = images + [im]

      concat_image = image_utils.concat_images(images, self._hparams.batch_size, im_shape)
      summaries.append(tf.summary.image(concat_image_name, concat_image, max_outputs=max_outputs))

    if show_all_separately:
      for image_name, image in self._image_dic.items():
        show_image(image_name, image)
    else:
      # show interest filter pipeline stages, for each scale
      for scale in self._hparams.scale_range:

        # 'positive features' fixations mask
        concat_images(concat_image_name=self.im_name_scale('ifn_core_positive', scale),
                      im_names=['image',
                                self.im_name_scale('features_pos', scale),
                                self.im_name_scale('non_maxima_pos', scale)])

        # 'negative features' fixations mask
        concat_images(concat_image_name=self.im_name_scale('ifn_core_negative', scale),
                      im_names=['image',
                                self.im_name_scale('features_neg', scale),
                                self.im_name_scale('non_maxima_neg', scale)])

      # culminated (reduced and sparsified) 'fixation_mask' + resulting 'positional encodings'
      # 'positive features'
      concat_images(concat_image_name='ifn_positive',
                    im_names=['conv_sum_pos', 'fixation_mask_pos'])

      # 'negative features'
      concat_images(concat_image_name='ifn_negative',
                    im_names=['conv_sum_neg', 'fixation_mask_neg'])

      # positive and negative combined
      concat_images(concat_image_name='ifn_fixmasks',
                    im_names=['fixation_mask_pos', 'fixation_mask_neg', 'fixation_mask'])

      # Positional Encoding is a bunch of filters, not square and hard to interpret
      # --------------------------------------------------------------------------------

      # show as square
      show_encs_as_square = False
      if show_encs_as_square:
        pos_image = self._image_dic['positional_encodings']
        shape = pos_image.get_shape().as_list()
        vol = np.prod(shape[1:])
        shape, _ = image_utils.square_image_shape_from_1d(vol)

        concat_images(concat_image_name='encodings_mask-pos',
                      im_names=['masked_encodings', 'positional_encodings'],
                      im_shape=shape)

      # show as reduced to one slice of the volume (i.e. all filters shown in one image)
      show_encs_reduced = True
      if show_encs_reduced:
        mask_image = self._image_dic['masked_encodings']
        mask_image_reduced = tf.reduce_max(mask_image, axis=3, keep_dims=True)    # across filters

        pos_image = self._image_dic['positional_encodings']
        pos_image_reduced = tf.reduce_max(pos_image, axis=3, keep_dims=True)    # across filters

        images = [mask_image_reduced, pos_image_reduced]
        concat_image = image_utils.concat_images(images, self._hparams.batch_size)
        summaries.append(tf.summary.image('reduced_encodings_mask-pos', concat_image, max_outputs=max_outputs))

    # show scalars
    for (name, op) in self.scalar_vals:
      summaries.append(tf.summary.scalar(name, op))

  def _depthwise_conv_with_2d_filter(self, in_tensor, filter2d):
    """Depthwise convolutional operation with 2D filter."""
    # convert 2d filter to appropriate shape: [filter_height, filter_width, in_channels, channel_multiplier
    channels = in_tensor.get_shape().as_list()[3]
    filter2d = filter2d[:, :, tf.newaxis, tf.newaxis]
    filter2d_shape = filter2d.get_shape().as_list()
    filter2d_newshape = [filter2d_shape[0], filter2d_shape[1], channels, 1]
    filter2d = tf.broadcast_to(filter2d, filter2d_newshape)

    # coarsely smooth the output to make insensitive to exact position of a feature
    out_tensor = tf.nn.depthwise_conv2d(in_tensor, filter2d, strides=[1, 1, 1, 1], padding='SAME')

    return out_tensor
