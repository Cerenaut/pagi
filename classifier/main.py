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

"""Classification framework for training and evaluating models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import tensorflow as tf

from experiment import Experiment
from classifier import Classifier


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_path', None, 'The path to dataset.')
tf.flags.DEFINE_string('model', 'logistic',
                       'Model to use for classification. Specify multiple '
                       'model types to build multiple classifiers, e.g. '
                       'logistic,svm')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'The name of the dataset to used in the experiment.')
tf.flags.DEFINE_string('svm_hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'svm hparams of this experiment.')
tf.flags.DEFINE_string('logistic_hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'logistic hparams of this experiment.')

tf.flags.DEFINE_integer('train_start', 0, 'Training set starting index.')
tf.flags.DEFINE_integer('train_len', 60000, 'Training set length.')
tf.flags.DEFINE_integer('test_start', 60000, 'Test set starting index.')
tf.flags.DEFINE_integer('test_len', 10000, 'Test set length.')
tf.flags.DEFINE_integer('seed', 42, 'Seed used for random shuffling.')

tf.flags.DEFINE_bool('learning_curve', False, 'Record learning curve.')
tf.flags.DEFINE_bool('shuffle', False, 'Shuffle the dataset prior to split.')
tf.flags.DEFINE_bool('verbose', False, 'Enable/disable verbose logging.')


def main(_):
  # Setup Experiment
  experiment = Experiment(FLAGS.dataset)

  try:
    # Load the exported data
    experiment.import_data(FLAGS.data_path)
  except OSError:
    logging.error(
        'Failed to load the exported data file: %s', FLAGS.data_path)
    os.abcprocess.exit(1)

  # Get features and labels
  features, labels, _ = experiment.get_data()

  # Validate data imported OK
  assert features is not None
  assert labels is not None

  # Parse the models to use (comma-separated)
  models = [x.strip() for x in FLAGS.model.split(',')]

  # Setup Classifier
  classifiers = {}
  for model in models:
    if model == 'svm':
      classifiers['svm'] = Classifier(
          'svm', FLAGS.svm_hparams_override)
    elif model == 'logistic':
      classifiers['logistic'] = Classifier(
          'logistic', FLAGS.logistic_hparams_override)
    else:
      raise NotImplementedError(model + 'is not implemented.')

  # Train/test split
  split_indices = {
      'train': (FLAGS.train_start, FLAGS.train_len),
      'test': (FLAGS.test_start, FLAGS.test_len)
  }

  # Training and evaluation
  results = {}
  for model in classifiers:
    results[model] = classifiers[model].classify(
        experiment, split_indices, verbose=FLAGS.verbose,
        seed=FLAGS.seed, shuffle=FLAGS.shuffle,
        record_learning_curve=FLAGS.learning_curve)

    print('==== ' + model + ' results ====\n')
    classifiers[model].classification_report(results[model])

    if FLAGS.learning_curve:
      classifiers[model].plot_learning_curve(
          filename=model + '_learning_curve.png')

if __name__ == '__main__':
  tf.app.run()
