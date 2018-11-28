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

"""Workflow base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import os

import tensorflow as tf

# Utilities
from classifier.harness import Harness
from utils import generic_utils as util
from utils import logger_utils
from utils.tf_utils import tf_label_filter


class Workflow(object):
  """Workflow base class."""

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        train_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        test_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        evaluate=True,
        train=True
    )

  def __init__(self, session, dataset_type, dataset_location, component_type, hparams_override, eval_opts, export_opts,
               opts=None, summarize=True, seed=None, summary_dir=None, checkpoint_opts=None):
    self._session = session

    self._dataset_type = dataset_type
    self._dataset_location = dataset_location

    self._component = None
    self._component_type = component_type
    self._hparams_override = hparams_override

    self._opts = opts.values()
    self._eval_opts = eval_opts
    self._export_opts = export_opts

    self._seed = seed
    self._summarize = summarize
    self._summary_dir = summary_dir

    self._last_step = 0
    self._freeze_training = False
    self._checkpoint_opts = checkpoint_opts

    self._operations = {}
    self._placeholders = {}

    self._setup()

  def _setup(self):
    """Setup the experiment"""

    # Get the model's default HParams
    self._hparams = self._component_type.default_hparams()

    # Override HParams from dict
    if self._hparams_override:
      self._hparams.override_from_dict(self._hparams_override)

    print('HParams:', json.dumps(self._hparams.values(), indent=4))
    logger_utils.log_param(self._hparams)

    # Setup dataset
    self._setup_dataset()

    # Setup model
    self._setup_component()

    # Setup saver to save/restore states
    self._setup_checkpoint_saver()

    # TensorBoard settings
    if self._summary_dir is None:
      self._summary_dir = util.get_summary_dir()
    logger_utils.log_param({'summary_dir': self._summary_dir})
    self._writer = tf.summary.FileWriter(self._summary_dir, self._session.graph)

    # Initialise variables in the graph
    if not self._checkpoint_opts['checkpoint_path']:
      init_op = tf.global_variables_initializer()
      self._session.run(init_op)

  def _setup_dataset(self):
    """Setup the dataset and retrieve inputs, labels and initializers"""
    with tf.variable_scope('dataset'):
      self._dataset = self._dataset_type(self._dataset_location)

      # Dataset for training
      train_dataset = self._dataset.get_train()
      # Filter training set to keep specified labels only
      train_classes = self._opts['train_classes']
      if train_classes and len(train_classes) > 0:
        train_dataset = train_dataset.filter(lambda x, y: tf_label_filter(x, y, self._opts['train_classes']))
      train_dataset = train_dataset.shuffle(buffer_size=10000)
      train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._hparams.batch_size))
      train_dataset = train_dataset.prefetch(1)
      train_dataset = train_dataset.repeat()  # repeats indefinitely

      # Evaluation dataset (i.e. no shuffling, pre-processing)
      eval_train_dataset = self._dataset.get_train()
      # Filter test set to keep specified labels only --> all evaluation (train and test)

      test_classes = self._opts['test_classes']
      if test_classes and len(test_classes) > 0:
        eval_train_dataset = eval_train_dataset.filter(lambda x, y: tf_label_filter(x, y, self._opts['test_classes']))
      eval_train_dataset = eval_train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._hparams.batch_size))
      eval_train_dataset = eval_train_dataset.prefetch(1)
      eval_train_dataset = eval_train_dataset.repeat(1)

      eval_test_dataset = self._dataset.get_test()
      # Filter test set to keep specified labels only --> all evaluation (train and test)
      if self._opts['test_classes']:
        eval_test_dataset = eval_test_dataset.filter(lambda x, y: tf_label_filter(x, y, self._opts['test_classes']))
      eval_test_dataset = eval_test_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._hparams.batch_size))
      eval_test_dataset = eval_test_dataset.prefetch(1)
      eval_test_dataset = eval_test_dataset.repeat(1)

      self._placeholders['dataset_handle'] = tf.placeholder(
          tf.string, shape=[], name='dataset_handle')

      # Setup dataset iterators
      with tf.variable_scope('dataset_iterators'):
        self._iterator = tf.data.Iterator.from_string_handle(self._placeholders['dataset_handle'],
                                                             train_dataset.output_types, train_dataset.output_shapes)
        self._inputs, self._labels = self._iterator.get_next()

        self._dataset_iterators = {}

        with tf.variable_scope('train_dataset'):
          self._dataset_iterators['training'] = train_dataset.make_initializable_iterator()

        with tf.variable_scope('eval_train_dataset'):
          self._dataset_iterators['eval_train'] = eval_train_dataset.make_initializable_iterator()

        with tf.variable_scope('eval_test_dataset'):
          self._dataset_iterators['eval_test'] = eval_test_dataset.make_initializable_iterator()

  def _setup_component(self):
    """Setup the component"""

    # Create the encoder component
    # -------------------------------------------------------------------------
    self._component = self._component_type()
    self._component.build(self._inputs, self._dataset.shape, self._hparams, 'component')

    if self._summarize:
      batch_types = ['training', 'encoding']
      if self._freeze_training:
        batch_types.remove('training')
      self._component.build_summaries(batch_types)  # Ask the component to unpack for you

  def _setup_checkpoint_saver(self):
    """Handles the saving and restoration of graph state and variables."""

    # Loads a subset of the checkpoint, specified by variable scopes
    if self._checkpoint_opts['checkpoint_load_scope']:
      load_scope = []
      init_scope = []

      scope_list = self._checkpoint_opts['checkpoint_load_scope'].split(',')
      for i in range(len(scope_list)):
        scope_list[i] = scope_list[i].lstrip().rstrip()

      global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

      for variable in global_variables:
        # Add any variables that match the specified scope to a separate list
        # Note: global_step is excluded and re-initialised, even if within scope
        if variable.name.startswith(tuple(scope_list)) and 'global_step' not in variable.name:
          load_scope.append(variable)
        else:
          init_scope.append(variable)

      # Load variables from specified scope
      saver = tf.train.Saver(load_scope)
      self._restore_checkpoint(saver, self._checkpoint_opts['checkpoint_path'])

      # Re-initialise any variables outside specified scope, including global_step
      init_scope_op = tf.variables_initializer(init_scope, name='init')
      self._session.run(init_scope_op)

      # Switch to encoding mode if freezing loaded scope
      if self._checkpoint_opts['checkpoint_frozen_scope']:
        self._freeze_training = True

      # Create new tf.Saver to save new checkpoints
      self._saver = tf.train.Saver()
      self._last_step = 0

    # Otherwise, attempt to load the entire checkpoint
    else:
      self._saver = tf.train.Saver()
      self._last_step = self._restore_checkpoint(self._saver, self._checkpoint_opts['checkpoint_path'])

  def _extract_checkpoint_step(self, path):
    """Returns the step from the file format name of TensorFlow checkpoints.

    Args:
      path: The checkpoint path returned by tf.train.get_checkpoint_state.
        The format is: {ckpnt_name}-{step}

    Returns:
      The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])

  def _restore_checkpoint(self, saver, checkpoint_path):
    """Loads an existing checkpoint, otherwises initialises."""
    prev_step = 0

    if checkpoint_path is not None:
      try:
        logging.info('Restoring checkpoint...')
        saver.restore(self._session, checkpoint_path)
        prev_step = self._extract_checkpoint_step(checkpoint_path)
      except ValueError:
        logging.error('Failed to restore parameters from %s', checkpoint_path)

    return prev_step

  def _on_before_training_batches(self):
    print('\n\n')
    logging.info('| Training Step | Loss          |   Batch  |   Epoch  |')
    logging.info('|---------------|---------------|----------|----------|')

  def _on_after_training_batch(self, batch, training_step, training_epoch):
    """
    Retrieves and records the loss after training one batch and prints result to the console.

    NOTE: This is currently too specific for the base Workflow class, as it can only work for an autoencoder.
    """
    # Get and record the loss for this batch
    loss = self._component.get_loss()
    logger_utils.log_metric({'mse': loss})

    # Output the results to console
    output = '| {0:>13} | {1:13.4f} | Batch {2}  | Epoch {3}  |'.format(training_step, loss, batch, training_epoch)
    logging.info(output)

  def helper_evaluate(self, batch):
    # Prepare inputs and run one batch
    self.evaluate(self._placeholders['dataset_handle'], self._dataset_iterators['eval_train'],
                  self._dataset_iterators['eval_test'], batch=batch)

  def run(self, num_batches, evaluate, train=True):
    """Run Experiment"""

    if train:
      training_handle = self._session.run(self._dataset_iterators['training'].string_handle())
      self._session.run(self._dataset_iterators['training'].initializer)

      self._on_before_training_batches()

      for batch in range(self._last_step, num_batches):

        training_step = self._session.run(tf.train.get_global_step(self._session.graph))
        training_epoch = self._dataset.get_training_epoch(self._hparams.batch_size, training_step)

        # Perform the training, and retrieve feed_dict for evaluation phase
        self.training(training_handle, batch)

        self._on_after_training_batch(batch, training_step, training_epoch)

        # Export any experiment-related data
        # -------------------------------------------------------------------------
        if self._export_opts['export_filters']:
          if (batch == num_batches-1) or ((batch + 1) % self._export_opts['interval_batches'] == 0):
            self.export(self._session)

        if self._export_opts['export_checkpoint']:
          if (batch == num_batches-1) or ((batch + 1) % num_batches == 0):
            self._saver.save(self._session, os.path.join(self._summary_dir, 'model.ckpt'), global_step=batch + 1)

        # evaluation: every N steps, test the encoding model
        if evaluate:
          # defaults to once per batches, and ensure at least once at the end
          if (batch == num_batches-1) or ((batch + 1) % self._eval_opts['interval_batches'] == 0):
            self.helper_evaluate(batch)
    elif evaluate:  # train is False
      self.helper_evaluate(0)
    else:
      logging.warning("Both 'train' and 'evaluate' flag are False, so nothing to run.")

  def _setup_train_feed_dict(self, batch_type, training_handle):
    feed_dict = {
        self._placeholders['dataset_handle']: training_handle
    }
    return feed_dict

  def _setup_train_batch_types(self):
    """Set the batch types, adjusting for which ones should be frozen"""
    batch_type = 'training'
    if self._freeze_training:
      batch_type = 'encoding'
    return batch_type

  def training(self, training_handle, training_step, training_fetches=None):
    """The training procedure within the batch loop"""

    if training_fetches is None:
      training_fetches = {}

    batch_type = self._setup_train_batch_types()
    feed_dict = self._setup_train_feed_dict(batch_type, training_handle)
    self.step_graph(self._component, feed_dict, batch_type, training_fetches)
    self._component.write_summaries(training_step, self._writer, batch_type=batch_type)

    return feed_dict

  def evaluate(self, dataset_handle, train_dataset_iter, test_dataset_iter, batch=None):
    """The evaluation procedure within the batch loop"""

    logging.info('Classifier starting...')

    classifier_model = self._eval_opts['model']
    classifier_hparams = self._eval_opts['hparams'][classifier_model]

    harness = Harness(classifier_model, classifier_hparams, None, self._labels, self._dataset.num_classes,
                      learning_curve=False, summary_dir=self._summary_dir, scale_features=self._eval_opts['unit_range'])

    harness.run(self._session, dataset_handle, train_dataset_iter, test_dataset_iter, self._component)
    results = harness.classify()
    best_result = max(results, key=lambda x: x['test']['accuracy'])

    logger_utils.log_metric({
        'train_accuracy': best_result['train']['accuracy'],
        'test_accuracy': best_result['test']['accuracy']
    })

    # Record the best test accuracy
    if self._summarize:
      summary = tf.Summary()
      summary.value.add(tag='classifier/train_accuracy', simple_value=best_result['train']['accuracy'])
      summary.value.add(tag='classifier/test_accuracy', simple_value=best_result['test']['accuracy'])
      self._writer.add_summary(summary, batch)
      self._writer.flush()

    harness.write_component_evaluation_summaries(self._component, batch, self._writer)

  def export(self, session):
    # Export all filters to disk
    self._component.write_filters(session, self._summary_dir)

  def step_graph(self, component, feed_dict, batch_type, fetches=None, is_update_feed_dict=True):
    """Encapsulate the stuff you need to do before and after a graph step: feed dict and fetches"""
    if fetches is None:
      fetches = {}
    if is_update_feed_dict:
      component.update_feed_dict(feed_dict, batch_type)
    component.add_fetches(fetches, batch_type)
    fetched = self._session.run(fetches, feed_dict=feed_dict)
    component.set_fetches(fetched, batch_type)
    return fetched
