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

"""Workflow base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import os

import tensorflow as tf

# Utilities
from pagi.utils import generic_utils as util
from pagi.utils import logger_utils
from pagi.utils.generic_utils import class_filter
from pagi.utils.tf_utils import tf_label_filter

from pagi.classifier.harness import Harness


class Workflow:
  """The base workflow for working with components."""

  @staticmethod
  def default_opts():
    """Builds an HParam object with default workflow options."""
    return tf.contrib.training.HParams(
        superclass=False,
        class_proportion=1.0,
        train_classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        test_classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        evaluate=False,
        validate=False,
        train=True,
        training_progress_interval=0,
        testing_progress_interval=0,
        profile_file=None
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

    self._run_options = None
    self._run_metadata = None

  def _test_consistency(self):
    """
    If multiple params relate to each other, make sure they are set consistently
    - prevent hard to diagnose runtime issues. This should be overrided in child workflows as necessary.
    """

  def setup(self):
    """Setup the experiment"""

    # Get the model's default HParams
    self._hparams = self._component_type.default_hparams()

    # Override HParams from dict
    if self._hparams_override:
      self._hparams.override_from_dict(self._hparams_override)

    hparams_json = self._hparams.to_json(indent=4)
    print('HParams:', hparams_json)
    logger_utils.log_param(self._hparams)

    # catch workflow/component setup issues before trying to setup component
    self._test_consistency()

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

    # Write all parameters to disk
    with open(os.path.join(self._summary_dir, 'params.json'), 'w') as f:
      params = {
          'hparams': self._hparams.values(),
          'workflow-options': self._opts
      }

      f.write(json.dumps(params, indent=4))

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
      # Filter dataset to keep specified classes only
      if self._opts['train_classes']:
        the_classes = class_filter(self._dataset, self._opts['train_classes'],
                                   self._opts['superclass'],
                                   self._opts['class_proportion'])
        train_dataset = train_dataset.filter(lambda x, y: tf_label_filter(x, y, the_classes))

      train_dataset = train_dataset.shuffle(buffer_size=10000)
      train_dataset = train_dataset.batch(self._hparams.batch_size, drop_remainder=True)
      train_dataset = train_dataset.prefetch(1)
      train_dataset = train_dataset.repeat()  # repeats indefinitely

      # Evaluation dataset (i.e. no shuffling, pre-processing)
      eval_train_dataset = self._dataset.get_train()
      # Filter test set to keep specified labels only --> all evaluation (train and test)
      if self._opts['test_classes']:
        the_classes = class_filter(self._dataset, self._opts['test_classes'],
                                   self._opts['superclass'],
                                   self._opts['class_proportion'])
        eval_train_dataset = eval_train_dataset.filter(lambda x, y: tf_label_filter(x, y, the_classes))
      eval_train_dataset = eval_train_dataset.batch(self._hparams.batch_size, drop_remainder=True)
      eval_train_dataset = eval_train_dataset.prefetch(1)
      eval_train_dataset = eval_train_dataset.repeat(1)

      eval_test_dataset = self._dataset.get_test()
      # Filter test set to keep specified labels only --> all evaluation (train and test)
      if self._opts['test_classes']:
        the_classes = class_filter(self._dataset, self._opts['test_classes'],
                                   self._opts['superclass'],
                                   self._opts['class_proportion'])
        eval_test_dataset = eval_test_dataset.filter(lambda x, y: tf_label_filter(x, y, the_classes))

      eval_test_dataset = eval_test_dataset.batch(self._hparams.batch_size, drop_remainder=True)
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
    self._component.build(self._inputs, self._dataset.shape, self._hparams)

    if self._summarize:
      batch_types = ['training', 'encoding']
      if self._freeze_training:
        batch_types.remove('training')
      self._component.build_summaries(batch_types)  # Ask the component to unpack for you

  def _setup_checkpoint_saver(self):
    """Handles the saving and restoration of graph state and variables."""

    max_to_keep = self._export_opts['max_to_keep']

    # Loads a subset of the checkpoint, specified by variable scopes
    if self._checkpoint_opts['checkpoint_load_scope']:
      load_scope = []
      init_scope = []

      scope_list = self._checkpoint_opts['checkpoint_load_scope'].split(',')
      for i, scope in enumerate(scope_list):
        scope_list[i] = scope.lstrip().rstrip()

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
      self._saver = tf.train.Saver(max_to_keep=max_to_keep)
      self._last_step = 0

    # Otherwise, attempt to load the entire checkpoint
    else:
      self._saver = tf.train.Saver(max_to_keep=max_to_keep)
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

  def _get_status(self):
    """Return some string proxy for the losses or errors being optimized"""
    loss = self._component.get_loss()
    return loss

  def _on_after_training_batch(self, batch, training_step, training_epoch):
    """
    Retrieves and records the loss after training one batch and prints result to the console.

    NOTE: This is currently too specific for the base Workflow class, as it can only work for an autoencoder.
    """
    # Get and record the loss for this batch
    status = self._get_status()
    logger_utils.log_metric({'status': status})

    # Output the results to console
    do_print = True
    training_progress_interval = self._opts['training_progress_interval']
    if training_progress_interval > 0:
      if (training_step % training_progress_interval) != 0:
        do_print = False

    if do_print:
      output = '| {0:>13} | {1:13.4f} | Batch {2}  | Epoch {3}  |'.format(training_step, status, batch, training_epoch)
      logging.info(output)

  def do_profile(self):
    file = self.get_profile_file()
    if file is None:
      return False
    return True

  def get_profile_file(self):
    file = self._opts['profile_file']
    return file

  def _setup_profiler(self):
    if self.do_profile():
      logging.info('Preparing profile info...')
      self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      self._run_metadata = tf.RunMetadata()

  def _run_profiler(self):
    if self.do_profile():
      logging.info('Writing profile info...')
      profile_file = 'timeline.json'

      # Create the Timeline object, and write it to a json
      tl = timeline.Timeline(self._run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open(profile_file, 'w') as f:
        f.write(ctf)

  def helper_evaluate(self, batch):
    # Prepare inputs and run one batch
    self.evaluate(self._placeholders['dataset_handle'], self._dataset_iterators['eval_train'],
                  self._dataset_iterators['eval_test'], batch=batch)

  def helper_validate(self, batch):
    # Prepare inputs and run one batch
    return

  def run(self, num_batches, evaluate, train=True):
    """Run Experiment"""
    validate = True  # This needs to be sorted somehow - where does this get called from?

    self._setup_profiler()

    global_step = 0
    phase_change = True

    if train:
      # TODO: The training handle needs to move into the training() method
      training_handle = self._session.run(self._dataset_iterators['training'].string_handle())
      self._session.run(self._dataset_iterators['training'].initializer)

      self._on_before_training_batches()

      for batch in range(self._last_step, num_batches):
        training_step = self._session.run(tf.train.get_global_step(self._session.graph))
        training_epoch = self._dataset.get_training_epoch(self._hparams.batch_size, training_step)
        global_step = batch

        # Perform the training, and retrieve feed_dict for evaluation phase
        self.training_step(training_handle, batch, phase_change)
        phase_change = False  # Ordinarily, don't reset

        self._on_after_training_batch(batch, training_step, training_epoch)

        # Export any experiment-related data
        # -------------------------------------------------------------------------
        if self._export_opts['export_filters']:
          if batch == (num_batches - 1) or (batch + 1) % self._export_opts['interval_batches'] == 0:
            self.export(self._session)

        if self._export_opts['export_checkpoint']:
          if batch == (num_batches - 1) or (batch + 1) % self._export_opts['interval_batches'] == 0:
            self._saver.save(self._session, os.path.join(self._summary_dir, 'model.ckpt'), global_step=batch + 1)

        # evaluation: every N steps, test the encoding model
        if validate:
          if (batch + 1) % self._eval_opts['interval_batches'] == 0:  # defaults to once per batches
            self.helper_validate(global_step)
            phase_change = True

      logging.info('Training & optional evaluation complete.')

      self._run_profiler()

    if evaluate:
      self.helper_evaluate(global_step)

  def session_run(self, fetches, feed_dict):
    if self.do_profile():
      logging.info('Running batch with profiler enabled.')

    return self._session.run(fetches, feed_dict=feed_dict, options=self._run_options, run_metadata=self._run_metadata)

  def _setup_train_feed_dict(self, batch_type, training_handle):
    del batch_type
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

  def export(self, session, feed_dict=None):
    del feed_dict

    # Export all filters to disk
    try:
      self._component.write_filters(session, self._summary_dir)
    except AttributeError as e:
      logging.warning('Failed to export filters.')
      logging.debug(e)

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
