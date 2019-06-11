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

"""Classifier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn import utils
from sklearn import metrics

from classifier.models.logistic_regression_model import LogisticRegressionModel
from classifier.models.svm_model import SvmModel


class Classifier(object):
  """
  Creates an instance of the classification model for training and inference,
  handles the preprocessing of the data and the allows easy access to the
  classification results.
  """

  def __init__(self, model, hparams_override=None, summary_dir='/tmp', name='classifier'):
    """
    Initialize the parameters of the classifier.

    Args:
        model: A string containing the name of classification model to use.
        hparams_override: A dict with specific hyperparameters for overriding.
        summary_dir: The location of classifier results.
        name: A unique name given to the classifier to diffrentiate it from others when using multiple classifiers.
    """
    self._model = model
    self._hparams_override = hparams_override
    self._summary_dir = summary_dir
    self._name = name

    self._results = {}
    self._hparams = None
    self._learning_curve = None

    self._log = logging.getLogger(self._name + '_logger')
    self._log_level = logging.DEBUG

  def classify(self, experiment, split_indices, seed=0, shuffle=False,
               verbose=False, record_learning_curve=False,
               scale_features=False):
    """
    Retrieves the dataset from an Experiment instance and prepares it for
    the classification model. Creates a new instance of the model in order
    to train then run inference on both training and test sets.

    Args:
        experiment: An instance of the Experiment class.
        split_indices: A dict containing tuples of the start index and
            length of both the training and test sets.
        seed: An integer used to specify the randomness seed.
        shuffle: A boolean to decide whether to shuffle the dataset prior
            to splitting into training and test sets.
        verbose: A boolean to enable verbose logging.
    Returns:
        results: A dict containing the results from inference on both the
            training and test sets. Results include the classification
            accuracy, confusion matrix and the F1 score.
    """
    # Enable logging
    if verbose:
      self._logging()

    features, labels, _ = experiment.get_data()
    features, labels = self._format_data(features, labels, scale_features)

    if shuffle:
      features, labels = utils.shuffle(
          features, labels, random_state=seed)

    x_train, x_test, y_train, y_test = self._train_test_split(
        features, labels, split_indices)

    # Flatten the data for sklearn
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    if verbose:
      logging.debug('Dataset shapes: ')
      logging.debug('x_train, y_train: %s, %s', x_train.shape, y_train.shape)
      logging.debug('x_test, y_test: %s, %s', x_test.shape, y_test.shape)

    # Choose the model
    if self._model == 'svm':
      self._hparams = SvmModel.default_hparams()
      if isinstance(self._hparams_override, dict):
        self._hparams.override_from_dict(self._hparams_override)
      elif self._hparams_override:
        self._hparams.parse(self._hparams_override)

      model = SvmModel(self._hparams, verbose=False)
    elif self._model == 'logistic':
      self._hparams = LogisticRegressionModel.default_hparams()
      if self._hparams_override and isinstance(self._hparams_override, dict):
        self._hparams.override_from_dict(self._hparams_override)
      elif self._hparams_override:
        self._hparams.parse(self._hparams_override)

      model = LogisticRegressionModel(self._hparams, verbose=False)
    else:
      raise NotImplementedError(self._model + 'is not implemented.')

    # Training
    model.train(x_train, y_train, seed=seed)

    # Learning Curves
    if record_learning_curve:
      self._learning_curve = model.learning_curve(features, labels)

    # Evaluation
    train_accuracy, train_preds = model.evaluate(x_train, y_train)
    test_accuracy, test_preds = model.evaluate(x_test, y_test)

    # Generate confusion matrices
    train_cm = metrics.confusion_matrix(y_train, train_preds)
    test_cm = metrics.confusion_matrix(y_test, test_preds)

    # Generate F1-score report
    train_f1 = metrics.classification_report(y_train, train_preds)
    test_f1 = metrics.classification_report(y_test, test_preds)

    # Record results
    self._results = {
        'train': {
            'f1_score': train_f1,
            'accuracy': train_accuracy,
            'confusion_matrix': train_cm
        },
        'test': {
            'f1_score': test_f1,
            'accuracy': test_accuracy,
            'confusion_matrix': test_cm
        }
    }

    return self._results

  def get_results(self):
    """
    Retrieves the results dict containing the results for both
    training and test sets.
    """
    return self._results

  def accuracy_score(self):
    """
    Retrieves the classification accuracy for both training and test sets.
    """
    return [
        self._results['train']['accuracy'],
        self._results['test']['accuracy']
    ]

  def confusion_matrix(self):
    """
    Retrieves the confusion matrix for both training and test sets.
    """
    return [
        self._results['train']['confusion_matrix'],
        self._results['test']['confusion_matrix']
    ]

  def f1_score(self):
    """Retrieves the F1 score for both training and test sets."""
    return [
        self._results['train']['f1_score'],
        self._results['test']['f1_score']
    ]

  def get_learning_curve(self):
    """
    Retrieves the recorded learning curve statistics which includes
    the training sizes, training scores and test scores.
    """
    return self._learning_curve

  def plot_learning_curve(self, filename='learning_curves.png',
                          title='Learning Curves'):
    """
    Plots the learning curves using the training and test scores. The
    matplotlib figure is automatically saved to disk.

    Returns:
        plt: The matplotlib figure.
    """
    if self._learning_curve is None:
      raise ValueError('Learning curve not recorded.')

    train_sizes, train_scores, test_scores = self._learning_curve

    plt.figure()
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Score')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label='Test score')

    plt.legend(loc='best')
    plt.savefig(filename)

    return plt

  def classification_report(self, results):
    """
    Prints out the contents of the results dictionary in a more readable
    and user friendly format.

    Args:
        results: A dict containing the results from inference on both the
            training and test sets. Results include the classification
            accuracy, confusion matrix and the F1 score.
    """
    self._log.info('Classifier HParams: %s\n', json.dumps(self._hparams.values(), indent=4))

    self._log.info('Training Accuracy: {0:f}%'.format(
        results['train']['accuracy'] * 100))
    self._log.info('Confusion Matrix')
    self._log.info('\n')
    self._log.info(results['train']['confusion_matrix'])
    self._log.info('F1 Score')
    self._log.info('\n')
    self._log.info(results['train']['f1_score'])

    self._log.info('\n')

    self._log.info('Test Accuracy: {0:f}%'.format(
        results['test']['accuracy'] * 100))
    self._log.info('Confusion Matrix')
    self._log.info(results['test']['confusion_matrix'])
    self._log.info('F1 Score')
    self._log.info(results['test']['f1_score'])

  def _format_data(self, features, labels, scale_features=False):
    """
    Preprocesses and formats the features and labels in the required format
    for the classification models.

    Args:
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing a label for each sample.
        scale_features: Option to re-scale features to a unit range
    Returns:
        features: A numpy n-dimensional array containing the flattened
            version of the features matrix with the following shape:
            [batch_size, num_features].
        labels: A numpy array containing the label for each sample.
    """
    # Flatten all dimensions except first dimension (batch)
    features = np.reshape(features, (-1, np.prod(features.shape[1:])))

    if scale_features:
      reciprocal_feature_max = 1.0 / features.max()
      features *= reciprocal_feature_max

    return features, labels

  def _logging(self):
    """Setup the logger to both print and save to file."""
    log_filename = '%s_%s_%s.log' % (
        datetime.datetime.now().strftime('%y%m%d-%H%M'), self._name, self._model)
    log_dir = os.path.join(self._summary_dir, 'results')

    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    # Logging Handlers
    formatter = logging.Formatter(
        '[%(filename)s:%(lineno)s - %(funcName)s() '
        '- %(levelname)s] %(message)s')

    fh = logging.FileHandler(os.path.join(log_dir, log_filename))
    fh.setLevel(self._log_level)
    fh.setFormatter(formatter)

    # Logger settings
    self._log.setLevel(self._log_level)
    self._log.addHandler(fh)

  def _train_test_split(self, features, labels, split_indices):
    """
    Splits the dataset (features, labels) into training and test sets
    using the provided split indices.

    Args:
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing the label for each sample.
        split_indices: A dict containing tuples that specify the starting
            index, and the length of the set for both training and test
            sets.
    Returns:
        Returns the split training and test features and labels in a format
        similar to the sklearn.model_selection.train_test_split module:
            x_train, x_test, y_train, y_test
    """
    train_start, train_len = split_indices['train']
    test_start, test_len = split_indices['test']

    train_end = train_start + train_len
    test_end = test_start + test_len

    return [
        features[train_start:train_end],
        features[test_start:test_end],
        labels[train_start:train_end],
        labels[test_start:test_end]
    ]
