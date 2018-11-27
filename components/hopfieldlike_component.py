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

"""HopfieldlikeComponent class."""

import sys
import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from components.component import Component
from utils import image_utils
from utils.dual import DualData

from utils.image_utils import add_square_as_square, square_image_shape_from_1d
from utils.layer_utils import activation_fn


class HopfieldlikeComponent(Component):
  """
  Hoppfield net inspired component. Main ideas of Hoppfield network are implemented.
  The training differs though, this version uses gradient descent to minimise the diff
  between input and output calculated simply by one pass through, no activation function.
  i.e.    Y = MX,   loss = Y - X

  Batch types are:
  - training  :  minimises the loss function for batch of inputs
  - encoding :  'retrieval' but use 'encoding' for compatibility other components. Recursive steps to produce output.

  Terminology:

  x_ext = external input
  x = input to Hopfield itself, computed from x_ext + x_fb
  z = weighted sum and bias, input to non-linearity
  y = output of net at time t
  x_fb = y(t-1)


  WARNINGS
  # Builds ONE 'recursive' summary subgraph (i.e. these may be produced in the context of any batch type)

  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        batch_size=1,
        learning_rate=0.0001,
        optimizer='adam',
        momentum=0.9,
        momentum_nesterov=False,
        use_feedback=True,
        memorise_method='pinv',  # pinv = pseudoinverse, otherwise tf optimisation
        nonlinearity='sigmoid'
    )

  def __init__(self):
    self._name = None
    self._hidden_name = None
    self._hparams = None
    self._dual = None
    self._input_shape = None
    self._input_values = None
    self._summary_values = None
    self._summary_recursive_values = None

  def reset(self):
    self._dual.get('y').set_values_to(0.0)

    loss = self._dual.get('loss')
    loss.set_values_to(0.0)

  @property
  def name(self):
    return self._name

  def get_loss(self):
    return self._dual.get_values('loss')

  def get_dual(self):
    return self._dual

  def get_decoding(self):
    """For consistency with other components, 'decoding' is output, y. Reshape to input dimensions."""
    return self._dual.get_values('decoding')

  def update_feed_dict(self, feed_dict, batch_type='training'):
    if batch_type == 'training':
      self.update_training_dict(feed_dict)
    if batch_type == 'encoding':
      self.update_encoding_dict(feed_dict)

  def _update_dict_fb(self, feed_dict):
    # set feedback from previous y output
    x_next = self._dual.get('y').get_values()
    x_fb = self._dual.get_pl('x_fb')
    feed_dict.update({
        x_fb: x_next
    })

  def update_feed_dict_input_gain_pl(self, feed_dict, gain):
    input_gain_pl = self._dual.get('input_gain').get_pl()
    feed_dict.update({
        input_gain_pl: [gain]
    })

  def add_fetches(self, fetches, batch_type='training'):
    if batch_type == 'training':
      self.add_training_fetches(fetches)
    if batch_type == 'encoding':
      self.add_encoding_fetches(fetches)

    summary_op = self._dual.get_op(HopfieldlikeComponent.summary_name(batch_type))
    if summary_op is not None:
      fetches[self._name]['summaries'] = summary_op

    summary_op = self._dual.get_op(HopfieldlikeComponent.summary_name('recursive'))
    if summary_op is not None:
      fetches[self._name]['summaries_recursive'] = summary_op

  def set_fetches(self, fetched, batch_type='training'):
    if batch_type == 'training':
      self.set_training_fetches(fetched)
    if batch_type == 'encoding':
      self.set_encoding_fetches(fetched)

    summary_op = self._dual.get_op(HopfieldlikeComponent.summary_name(batch_type))
    if summary_op is not None:
      self._summary_values = fetched[self._name]['summaries']

    summary_recursive_op = self._dual.get_op(HopfieldlikeComponent.summary_name('recursive'))
    if summary_recursive_op is not None:
      self._summary_recursive_values = fetched[self._name]['summaries_recursive']

  def build_summaries(self, batch_types=None, scope=None):
    """Builds all summaries."""
    if not scope:
      scope = self._name + '/summaries/'
    with tf.name_scope(scope):
      for batch_type in batch_types:

        # build 'batch_type' summary subgraph
        with tf.name_scope(batch_type):
          summaries = self._build_summaries(batch_type)
          if summaries:
            self._dual.set_op(HopfieldlikeComponent.summary_name(batch_type), tf.summary.merge(summaries))

      # WARNING: Build ONE 'recursive' summary subgraph (i.e. these may be produced in the context of any batch type)
      with tf.name_scope('recursive'):
        summaries = self._build_recursive_summaries()
        self._dual.set_op(HopfieldlikeComponent.summary_name('recursive'), tf.summary.merge(summaries))

  def write_summaries(self, step, writer, batch_type='training'):
    """Write the summaries fetched into _summary_values"""
    if self._summary_values is not None:
      writer.add_summary(self._summary_values, step)
      writer.flush()

  def write_recursive_summaries(self, step, writer, batch_type='training'):
    """
    This is here for backward compatibility with the harness,
    We just write all recursive loops to the summary for whichever batch_type we're in for now.

    Only write summaries for encoding batch_type (retrieval)
    """

    if batch_type == 'encoding':
      if self._summary_recursive_values is not None:
        writer.add_summary(self._summary_recursive_values, step)
        writer.flush()

# ---------------- build methods

  def build(self, input_values, input_shape, hparams, name='hopfield_like'):
    """Builds the network and optimiser."""
    self._input_values = input_values
    self._input_shape = input_shape
    self._hparams = hparams
    self._name = name

    self._dual = DualData(self._name)

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      self._build()

      if self._is_pinv():
        self._build_memorise_pinv()
      elif self._is_pinv_hybrid():
        self._build_memorise_pinv_hybrid()
      else:
        self._build_memorise_optimizer()

      self.reset()

  def _setup_input(self):
    """Prepare input by reshaping and optionally applying an input gain."""
    input_values_shape = self._input_values.get_shape().as_list()
    input_size = np.prod(input_values_shape[1:])
    input_shape = [self._hparams.batch_size, input_size]

    x_ext = tf.reshape(self._input_values, input_shape)

    # apply input gain (can be used to amplify or attenuate external input)
    input_gain = self._dual.add('input_gain', shape=[1], default_value=1.0).add_pl(default=True)
    x_ext_gain = tf.multiply(x_ext, input_gain)

    # placeholder for getting feedback signal
    x_fb = self._dual.add('x_fb', shape=input_shape, default_value=0.0).add_pl(default=True)
    x = x_ext_gain + x_fb

    return x

  def b_variable(self, shape, trainable=False):
    b_default = -0.5  # shift sigmoid so that it is centred on 0.5 (~ >0.5=1, <0.5=0)
    b_initializer = tf.constant_initializer(b_default)

    if not self._is_pinv():
      trainable = True
      b_initializer = tf.zeros_initializer

    return tf.get_variable(name='b', shape=shape, initializer=b_initializer, trainable=trainable)

  def k_variable(self, shape, trainable=False):
    k_default = 10.0
    k_initializer = tf.constant_initializer(k_default)

    if not self._is_pinv():
      trainable = True
      k_initializer = tf.ones_initializer

    return tf.get_variable(name='k', shape=shape, initializer=k_initializer, trainable=trainable)

  def w_variable(self, shape, trainable=False):
    w_default = 0.01
    w_initializer = w_default * tf.random_uniform(shape)

    if not self._is_pinv() and not self._is_pinv_hybrid():
      trainable = True

    # Apply a constraint to zero out single cell circular weights (i.e. cell 1 to cell 1)
    return tf.get_variable(name='w', initializer=w_initializer, constraint=self._remove_diagonal,
                           trainable=trainable)

  def _build(self):
    """Initialises variables and builds the network."""
    input_values_shape = self._input_values.get_shape().as_list()
    input_size = np.prod(input_values_shape[1:])

    x = self._setup_input()

    # create variables
    b = self.b_variable(shape=[1, input_size])
    k = self.k_variable(shape=[1, input_size])
    w = self.w_variable(shape=[input_size, input_size])

    # setup network
    z = k * (tf.matmul(x, w) + b)         # weighted sum + bias
    y, _ = activation_fn(z, self._hparams.nonlinearity)    # non-linearity

    # calculate Hopfield Energy
    e = -0.5 * tf.matmul(tf.matmul(x, w), tf.transpose(x)) - tf.matmul(x, tf.transpose(b))

    # calculate loss (mean square error between input and output, like reconstruction loss)
    # WARNING!!! - I don't know if this is before or after the new values have been assigned by `_build_memorise_pinv`
    loss = tf.reduce_mean(tf.square(x - y))

    # 'decoding' for output in same dimensions as input, and for consistency with other components
    y_reshaped = tf.reshape(y, input_values_shape)

    # remember values for later use
    self._dual.set_op('x', x)
    self._dual.set_op('w', w)
    self._dual.set_op('b', b)
    self._dual.set_op('k', k)
    self._dual.set_op('y', y)
    self._dual.set_op('decoding', y_reshaped)
    self._dual.set_op('e', e)
    self._dual.set_op('loss', loss)

    return y

  def _build_memorise_pinv_hybrid(self):
    self._build_memorise_pinv()
    self._build_memorise_optimizer()

  def _build_memorise_pinv(self):
    """Pseudoinverse-based optimisation."""
    input_values_shape = self._input_values.get_shape().as_list()
    input_size = np.prod(input_values_shape[1:])

    x = self._dual.get_op('x')

    batches = input_values_shape[0]
    x_matrix = tf.reshape(x, [1, batches, input_size])  # 1 matrix of x vol vecs (expressed as 1 batch)
    xinv = tfp.math.pinv(x_matrix, rcond=None, validate_args=False, name=None)  # this is XT-1 (transposed already)
    w_batches = tf.matmul(xinv, x_matrix)
    w_val = tf.reshape(w_batches, [input_size, input_size])   # strip out the batch dimension

    w_ref = self._dual.get_op('w')
    w = tf.assign(w_ref, w_val, name='w_assign')

    y_memorise = tf.matmul(x, w)
    loss_memorise = tf.reduce_sum(tf.square(x - y_memorise))

    self._dual.set_op('y_memorise', y_memorise)
    self._dual.set_op('loss_memorise', loss_memorise)

  def _build_memorise_optimizer(self):
    """Minimise loss using initialised a tf.train.Optimizer."""
    loss = self._dual.get_op('loss')

    optimizer = self._setup_optimizer()
    training = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

    self._dual.set_op('training', training)

  def _setup_optimizer(self):
    """Initialise the Optimizer class specified by a hyperparameter."""
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

  # ---------------- helpers

  @staticmethod
  def _remove_diagonal(tensor):
    mask = np.ones(tensor.get_shape(), dtype=np.float32)
    np.fill_diagonal(mask, 0)
    diagonal_mask = tf.convert_to_tensor(mask)
    weights_updated = tf.multiply(tensor, diagonal_mask)  # must be element-wise
    return weights_updated

  def _is_pinv(self):
    return self._hparams.memorise_method == 'pinv'

  def _is_pinv_hybrid(self):
    return self._hparams.memorise_method == 'pinv_hybrid'

  # ---------------- training

  def update_training_dict(self, feed_dict):
    pass

  def add_training_fetches(self, fetches):
    if self._is_pinv():
      names = ['loss_memorise', 'y_memorise', 'x', 'w', 'b', 'k']
    elif self._is_pinv_hybrid():
      names = ['loss_memorise', 'y_memorise', 'loss', 'training', 'y', 'w']
    else:
      names = ['loss', 'training', 'y', 'w']
    self._dual.add_fetches(fetches, names)

  def set_training_fetches(self, fetched):
    if self._is_pinv():
      names = ['loss_memorise', 'y_memorise', 'x', 'w', 'b', 'k']
    elif self._is_pinv_hybrid():
      names = ['loss_memorise', 'y_memorise', 'loss', 'y']
    else:
      names = ['loss', 'y']
    self._dual.set_fetches(fetched, names)

# ---------------- inference (encoding)

  def update_encoding_dict(self, feed_dict):
    self._update_dict_fb(feed_dict)

  def add_encoding_fetches(self, fetches):
    if self._is_pinv():
      names = ['decoding', 'y', 'x', 'w', 'b', 'k']
    else:
      names = ['loss', 'decoding', 'y']
    self._dual.add_fetches(fetches, names)

  def set_encoding_fetches(self, fetched):
    if self._is_pinv():
      names = ['decoding', 'y', 'x', 'w', 'b', 'k']
    else:
      names = ['loss', 'decoding', 'y']
    self._dual.set_fetches(fetched, names)

# -------------- build summaries

  @staticmethod
  def summary_name(batch_type):
    return 'summary_' + batch_type

  def _build_summaries(self, batch_type='training'):
    """Assumes appropriate name scope has been set."""
    summaries = []
    if batch_type == 'training':
      summaries = self._build_summaries_memorise(summaries, verbose=True)
    if batch_type == 'encoding':
      summaries = self._build_summaries_retrieve(summaries, verbose=True)
    return summaries

  def _build_recursive_summaries(self):
    """
    Assumes appropriate name scope has been set. Same level as _build_summaries.

    Build same summaries as retrieval.
    """
    summaries = []
    summaries = self._build_summaries_retrieve(summaries, verbose=True)
    return summaries

  def _build_summaries_retrieve(self, summaries, verbose=False):
    """Build summaries for retrieval."""
    max_outputs = 3
    x = self._dual.get_op('x')
    y = self._dual.get_op('y')
    w = self._dual.get_op('w')

    with tf.name_scope('vars'):
      # images
      add_square_as_square(summaries, w, 'w')

    with tf.name_scope('io'):
      summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)

      if verbose:
        # external input
        x_ext = self._input_values
        x_ext_reshaped = tf.reshape(x_ext, summary_input_shape)
        summaries.append(tf.summary.image('x_ext', x_ext_reshaped, max_outputs=max_outputs))

        # feedback signal (should be the same as the previous output)
        x_fb = self._dual.get_pl('x_fb')
        x_fb_reshaped = tf.reshape(x_fb, summary_input_shape)
        summaries.append(tf.summary.image('x_fb', x_fb_reshaped, max_outputs=max_outputs))

      # input to the net (not to the component, which is scaled and added to the fb before it becomes 'x')
      x_reshape = tf.reshape(x, summary_input_shape)
      summaries.append(tf.summary.image('x', x_reshape, max_outputs=max_outputs))

      # output of the net
      y_reshape = tf.reshape(y, summary_input_shape)
      summaries.append(tf.summary.image('y', y_reshape, max_outputs=max_outputs))

    with tf.name_scope('performance'):
      e = self._dual.get_op('e')
      summaries.append(tf.summary.scalar('Energy', tf.reduce_sum(e)))

      loss = self._dual.get_op('loss')
      summaries.append(tf.summary.scalar('Loss', tf.reduce_sum(loss)))

    return summaries

  def _build_summaries_memorise(self, summaries, verbose=False):
    """Build summaries for memorisation."""
    max_outputs = 3
    w = self._dual.get_op('w')
    k = self._dual.get_op('k')
    b = self._dual.get_op('b')
    loss = self._dual.get_op('loss')
    loss_memorise = self._dual.get_op('loss_memorise')
    x = self._dual.get_op('x')
    y = self._dual.get_op('y')
    y_memorise = self._dual.get_op('y_memorise')

    with tf.name_scope('io'):
      summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)

      x_reshape = tf.reshape(x, summary_input_shape)
      summaries.append(tf.summary.image('x', x_reshape, max_outputs=max_outputs))

      if y is not None and not self._is_pinv():
        # doesn't mean anything on pinv, where only one iteration of training, and input was indeterminate
        y_reshape = tf.reshape(y, summary_input_shape)
        summaries.append(tf.summary.image('y', y_reshape, max_outputs=max_outputs))

      if y_memorise is not None:
        y_reshape = tf.reshape(y_memorise, summary_input_shape)
        summaries.append(tf.summary.image('y_memorise', y_reshape, max_outputs=max_outputs))

    with tf.name_scope('vars'):
      # images
      add_square_as_square(summaries, w, 'w')

      b_size = np.prod(b.get_shape().as_list())
      b_shape, _ = square_image_shape_from_1d(b_size)
      b_reshaped = tf.reshape(b, b_shape)
      summaries.append(tf.summary.image('b', b_reshaped, max_outputs=max_outputs))

      k_size = np.prod(k.get_shape().as_list())
      k_shape, _ = square_image_shape_from_1d(k_size)
      k_reshaped = tf.reshape(k, k_shape)
      summaries.append(tf.summary.image('k', k_reshaped, max_outputs=max_outputs))

      # Monitor parameter sum over time
      with tf.name_scope('sum'):
        w_sum_summary = tf.summary.scalar('w', tf.reduce_sum(tf.abs(w)))
        b_sum_summary = tf.summary.scalar('b', tf.reduce_sum(tf.abs(b)))
        k_sum_summary = tf.summary.scalar('k', tf.reduce_sum(tf.abs(k)))
        summaries.extend([w_sum_summary, b_sum_summary, k_sum_summary])

      # histograms
      with tf.name_scope('hist'):
        summaries.append(tf.summary.histogram('w', w))
        summaries.append(tf.summary.histogram('b', b))
        summaries.append(tf.summary.histogram('k', k))

    with tf.name_scope('opt'):
      if loss is not None and not self._is_pinv():
        # doesn't mean anything on pinv, where only one iteration of training, and input was indeterminate
        summaries.append(tf.summary.scalar('loss', loss))

      if loss_memorise is not None:
        summaries.append(tf.summary.scalar('loss_memorise', loss_memorise))

    return summaries
