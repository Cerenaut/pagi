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

"""AutoencoderComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils
from pagi.utils.np_utils import np_write_filters
from pagi.utils.layer_utils import activation_fn
from pagi.utils.tf_utils import tf_build_stats_summaries
from pagi.components.summarize_levels import SummarizeLevels
from pagi.components.summary_component import SummaryComponent


class AutoencoderComponent(SummaryComponent):
  """
  Fully Connected Autoencoder with tied weights (and untied biases).
  It has the feature that you can mask the hidden layer, to apply arbitrary
  sparsening, or implement dropout.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        learning_rate=0.005,
        loss_type='mse',
        encoder_nonlinearity='none',
        decoder_nonlinearity='none',
        batch_size=64,
        filters=1024,
        optimizer='adam',
        momentum=0.9,
        momentum_nesterov=False,
        secondary=True,
        use_bias=True,  # Use bias in the encoding weighted sum
        summarize_level=SummarizeLevels.ALL.value,
        summarize_encoding=False,
        summarize_decoding=False,
        summarize_input=False,
        summarize_weights=False,
        max_outputs=3   # Number of outputs in TensorBoard
    )

  def build(self, input_values, input_shape, hparams, name='autoencoder', encoding_shape=None):
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape: The shape of the input, for display (internal is vectorized)
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
        encoding_shape: The shape to be used to display encoded (hidden layer) structures
    """
    self._input_values = input_values
    self._input_shape = input_shape
    self._hparams = hparams
    self.name = name
    self._encoding_shape = encoding_shape

    if self._encoding_shape is None:
      self._encoding_shape = self._create_encoding_shape_4d(input_shape)

    self._weights = None
    self._hidden_name = 'hidden'

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      self._build()
      self._build_optimizer()

    self.reset()

  def _create_encoding_shape_4d(self, input_shape):  # pylint: disable=W0613
    """Put it into convolutional geometry: [batches, filter h, filter w, filters]"""
    return [self._hparams.batch_size, 1, 1, self._hparams.filters]

  def _hidden_image_summary_shape(self):
    """image shape of hidden layer for summary."""
    volume = np.prod(self._encoding_shape[1:])
    square_image_shape, _ = image_utils.square_image_shape_from_1d(volume)
    return square_image_shape

  def get_batch_type(self):
    return self._dual.get_pl('batch_type')

  def get_loss(self):
    return self._loss

  def _build(self):
    """Build the autoencoder network"""

    batch_type_pl = self._dual.add('batch_type', shape=[], default_value='training').add_pl(
        default=True, dtype=tf.string)

    with tf.name_scope('encoder'):
      input_shape = self._input_values.get_shape().as_list()

      secondary_encoding_input = self._dual.add(
          'secondary_encoding_input', shape=input_shape, default_value=1.0).add_pl(default=True)

      # if batch_type == secondary_encoding: secondary_encoding_input
      # else: self._input_values
      encoding_input = tf.cond(
          tf.equal(batch_type_pl, 'secondary_encoding'),
          lambda: secondary_encoding_input,
          lambda: self._input_values
      )

      training_encoding, testing_encoding = self._build_encoding(encoding_input)
      training_filtered, testing_filtered = self._build_filtering(training_encoding, testing_encoding)

      # Inference output fork, doesn't accumulate gradients
      stop_gradient = tf.stop_gradient(testing_filtered)  # this is the encoding op

      self._dual.set_op('encoding', stop_gradient)
      self._dual.set_op('zs', training_filtered)

    with tf.name_scope('decoder'):
      filtered_shape = training_filtered.get_shape().as_list()

      secondary_decoding_input = self._dual.add(
          'secondary_decoding_input', shape=filtered_shape, default_value=1.0).add_pl(default=True)

      # if batch_type == training: training_filtered
      # elif batch_type == secondary_decoding: secondary_decoding_input
      # else: stop_gradient
      filtered_encoding = tf.cond(
          tf.equal(batch_type_pl, 'training'),
          lambda: training_filtered,
          lambda: tf.cond(
              tf.equal(batch_type_pl, 'secondary_decoding'),
              lambda: secondary_decoding_input,
              lambda: stop_gradient
          )
      )

      # Add decoding gain to filtered encoding
      decode_gain = self._dual.add('decode_gain', shape=[1], default_value=1.0).add_pl(default=True)
      filtered_encoding_with_gain = filtered_encoding * decode_gain

      decoding = self._build_decoding(self._hidden_name, self._input_values.get_shape().as_list(),
                                      filtered_encoding_with_gain)
      self._dual.set_op('decoding', decoding)

  def _build_weighted_sum(self, input_tensor, add_bias=True, trainable=True):
    """Compute the weighted sum using a Dense layer."""
    input_shape = input_tensor.get_shape().as_list()
    input_area = np.prod(input_shape[1:])

    hidden_size = self._hparams.filters
    hidden_name = self._hidden_name

    batch_input_shape = (-1, input_area)

    # Input reshape: Ensure flat (vector) x batch size input
    # -----------------------------------------------------------------
    name = hidden_name + '_input'
    input_vector = tf.reshape(input_tensor, batch_input_shape, name=name)
    logging.debug(input_vector)

    # Compute the weighted sum
    # -----------------------------------------------------------------
    weighted_sum = tf.layers.dense(inputs=input_vector,
                                   units=hidden_size,
                                   activation=None,       # Note we use our own non-linearity, elsewhere.
                                   use_bias=add_bias,
                                   name=hidden_name,
                                   trainable=trainable)

    with tf.variable_scope(hidden_name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
      if self._weights is None:
        self._weights = tf.get_variable('kernel')

    return weighted_sum

  def _build_encoding(self, input_tensor, mask_pl=None):
    """Build the encoder network"""

    hidden_size = self._hparams.filters
    batch_hidden_shape = (self._hparams.batch_size, hidden_size)

    weighted_sum = self._build_weighted_sum(input_tensor, self._hparams.use_bias)

    # Nonlinearity
    # -----------------------------------------------------------------
    hidden_transfer, _ = activation_fn(weighted_sum, self._hparams.encoder_nonlinearity)

    # External masking
    # -----------------------------------------------------------------
    name = self._hidden_name + '_masked'
    mask_pl = self._dual.add('mask', shape=batch_hidden_shape, default_value=1.0).add_pl()
    hidden_masked = tf.multiply(hidden_transfer, mask_pl, name=name)
    return hidden_masked, hidden_masked

  def _build_filtering(self, training_encoding, testing_encoding):
    return training_encoding, testing_encoding  # no hidden filtering, by default

  def _build_decoding(self, hidden_name, decoding_shape, filtered):
    """
    Build a decoder in feedback path to reconstruct the feedback input (i.e. previous hidden state)
    :param hidden_name the name of the dense layer from which to extract weights
    :param decoding_shape: shape of the reconstruction
    :param filtered: the hidden state to be used to reconstructed

    """

    input_area = np.prod(decoding_shape[1:])
    hidden_activity = filtered

    # Decoding: Reconstruction of the input
    # -----------------------------------------------------------------
    with tf.variable_scope(hidden_name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
      decoding_bias = tf.get_variable(name='decoding_bias', shape=[input_area],
                                      initializer=tf.zeros_initializer)

      weights = tf.get_variable('kernel')
      decoding_weighted_sum = tf.matmul(hidden_activity, weights, transpose_b=True,
                                        name=(hidden_name + '/decoding_weighted_sum'))
      decoding_biased_sum = tf.nn.bias_add(decoding_weighted_sum, decoding_bias)

      decoding_transfer, _ = activation_fn(decoding_biased_sum, self._hparams.decoder_nonlinearity)
      decoding_reshape = tf.reshape(decoding_transfer, decoding_shape)
      return decoding_reshape

  def _setup_optimizer(self):
    """Initialise the optimizer class depending on specified option (e.g. adam, momentum, sgd)."""
    if self._hparams.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(self._hparams.learning_rate)
    elif self._hparams.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(self._hparams.learning_rate, self._hparams.momentum,
                                             use_nesterov=self._hparams.momentum_nesterov)
    elif self._hparams.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self._hparams.learning_rate)
    else:
      raise NotImplementedError('Optimizer not implemented: ' + str(self._hparams.optimizer))

    return optimizer

  def _build_optimizer(self):
    """Setup the training operations"""
    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):

      self._optimizer = self._setup_optimizer()

      original = self._input_values
      decoding = self.get_decoding_op()
      loss = self._build_loss_fn(original, decoding)
      training_op = self._optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

      self._dual.set_op('loss', loss)
      self._dual.set_op('training', training_op)

  def _build_loss_fn(self, target, output):
    if self._hparams.loss_type == 'mse':
      return tf.losses.mean_squared_error(target, output)

    raise NotImplementedError('Loss function not implemented: ' + str(self._hparams.loss_type))

  # OP ACCESS ------------------------------------------------------------------
  def get_encoding_op(self):
    return self.get_op('encoding')

  def get_decoding_op(self):
    return self.get_op('decoding')

  def get_encoding(self):
    return self.get_values('encoding')

  def get_decoding(self):
    return self.get_values('decoding')

  def get_features(self, batch_type='training'):  # pylint: disable=W0613
    return self.get_encoding()

  # MODULAR INTERFACE ------------------------------------------------------------------
  def reset(self):
    """Reset the trained/learned variables and all other state of the component to a new random state."""
    self._loss = 0.0

    # TODO: This should re-initialize all the variables.

  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Updates the feed dict."""

    names = ['mask']
    self._dual.update_feed_dict(feed_dict, names)

    feed_dict.update({
        self._dual.get_pl('batch_type'): batch_type
    })

  def add_fetches(self, fetches, batch_type='training'):
    """Adds ops that will get evaluated."""
    names = ['loss', 'encoding', 'decoding']

    if batch_type == 'training':
      names.extend(['training'])

    self._dual.add_fetches(fetches, names)

    # Summaries
    super().add_fetches(fetches, batch_type)

  def set_fetches(self, fetched, batch_type='training'):
    """Store updated tensors"""

    # Loss (not a tensor)
    self._loss = fetched[self.name]['loss']

    names = ['encoding', 'decoding']
    self._dual.set_fetches(fetched, names)

    # Summaries
    super().set_fetches(fetched, batch_type)

  def _build_summaries(self, batch_type=None, max_outputs=3):
    """Builds all summaries."""
    summaries = []
    max_outputs = self._hparams.max_outputs

    if self._hparams.summarize_level == SummarizeLevels.OFF.value:
      return summaries

    encoding_op = self.get_encoding_op()
    decoding_op = self.get_decoding_op()

    if self._hparams.summarize_encoding:
      summaries.append(self._summary_hidden(encoding_op, 'encoding', max_outputs))

    if self._hparams.summarize_decoding:
      summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)
      input_summary_reshape = tf.reshape(self._input_values, summary_input_shape)

      # Show input on it's own
      input_alone = True
      if input_alone:
        summaries.append(tf.summary.image('input', input_summary_reshape, max_outputs=max_outputs))

      # Concatenate input and reconstruction summaries
      decoding_summary_reshape = tf.reshape(decoding_op, summary_input_shape)
      summary_reconstruction = tf.concat([input_summary_reshape, decoding_summary_reshape], axis=1)
      reconstruction_summary_op = tf.summary.image('reconstruction', summary_reconstruction,
                                                   max_outputs=max_outputs)
      summaries.append(reconstruction_summary_op)

    if self._hparams.summarize_weights:
      summaries.append(tf.summary.histogram('weights', self._weights))
      summaries.append(tf.summary.histogram('encoding', self._dual.get_op('encoding')))

    if self._hparams.summarize_input:
      input_stats_summary = tf_build_stats_summaries(self._input_values, 'input-stats')
      summaries.append(input_stats_summary)

    # Loss
    loss_summary = tf.summary.scalar('loss', self._dual.get_op('loss'))
    summaries.append(loss_summary)

    return summaries

  def _summary_hidden(self, hidden, name, max_outputs=3):
    """Return a summary op of a 'square as possible' image of hidden, the tensor for the hidden state"""
    hidden_shape_4d = self._hidden_image_summary_shape()  # [batches, height=1, width=filters, 1]
    summary_reshape = tf.reshape(hidden, hidden_shape_4d)
    summary_op = tf.summary.image(name, summary_reshape, max_outputs=max_outputs)
    return summary_op

  def write_filters(self, session, folder=None):
    """Write the learned filters to disk."""
    hidden_name = self._hidden_name

    with tf.variable_scope(self.name + '/' + hidden_name, reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):

      if self._weights is None:
        self._weights = tf.get_variable('kernel')
        logging.debug('weights: %s', self._weights)  # shape=(784, 1024) = input, cells

      weights_values = session.run([self._weights])
      weights_transpose = np.transpose(weights_values)

      filter_height = self._input_shape[1]
      filter_width = self._input_shape[2]

      file_name = "filters_" + self.name + ".png"
      if folder is not None and folder != "":
        file_name = folder + '/' + file_name

      np_write_filters(weights_transpose, [filter_height, filter_width], file_name=file_name)
