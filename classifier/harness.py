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

"""Harness class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from classifier.classifier import Classifier
from classifier.experiment import Experiment


class Harness(object):
  """
  Evaluates features/labels and performs classification using the Classifier Component.
  """

  def __init__(self, model, hparams, images, labels, num_classes, learning_curve=False, summary_dir='/tmp',
               scale_features=False, keep_images=False):
    """Initializes the model parameters."""
    self._model = model
    self._images = images
    self._labels = labels
    self._num_classes = num_classes
    self._learning_curve = learning_curve
    self._summary_dir = summary_dir
    self._scale_features = scale_features
    self._keep_images = keep_images

    self._hparams = ''
    self._hparams_search = {}

    self._experiment = None

    # Store hparams to search separately if provided in a list
    for hparam_key, hparam_value in hparams.items():
      if isinstance(hparam_value, list):
        self._hparams_search[hparam_key] = hparam_value
      else:
        self._hparams = '{0}={1},'.format(hparam_key, hparam_value)

  def reset(self):
    # TODO We should encapsulate this into a dict or someth with keys = {'train', 'test'} etc.
    self.train_features = self.train_labels = self.train_images = None
    self.test_features = self.test_labels = self.test_images = None
    self._experiment = None

  def append(self, batch_features, batch_labels, is_training=True, batch_images=None):
    if is_training:
      self.train_features, self.train_labels, self.train_images = self._append(
          batch_features, batch_labels, batch_images, self.train_features, self.train_labels, self.train_images)
    else:
      self.test_features, self.test_labels, self.test_images = self._append(
          batch_features, batch_labels, batch_images, self.test_features, self.test_labels, self.test_images)

  def _append(self, batch_features, batch_labels, batch_images, output_features, output_labels, output_images):
    """Appends evaluated features, labels and images to their respective output variables."""
    # Concatenate features and labels
    if output_features is None and output_labels is None:
      output_features = np.copy(batch_features)
      output_labels = np.copy(batch_labels)
    else:
      output_features = np.concatenate(
          [output_features, batch_features], axis=0)
      output_labels = np.concatenate(
          [output_labels, batch_labels], axis=0)

    # Images are optional, because they're big
    if batch_images is not None:
      if output_images is None:
        output_images = np.copy(batch_images)
      else:
        output_images = np.concatenate(
            [output_images, batch_images], axis=0)

    return output_features, output_labels, output_images

  def classify(self):
    """Perform the classification"""
    # Calculate the indices for the classifier
    split_indices = {
        'train': (0, self.train_features.shape[0]),
        'test': (self.train_features.shape[0], self.test_features.shape[0])
    }
    logging.debug('Indices: %s', str(split_indices))

    return self._classify(self._experiment, split_indices)

  def run(self, session, handle_pl, train_iterator, test_iterator, feature_detector):
    """
    Perform the supervised operations:
      1) Evaluate training & test data
      2) Perform classification using the evaluated data
    """
    self.reset()

    # Evaluate training set
    logging.info('Evaluating the training set...')
    eval_train_handle = session.run(train_iterator.string_handle())
    session.run(train_iterator.initializer)

    batch = 0
    while True:
      try:
        logging.info('Classifier Training Batch #%s', batch)
        batch += 1
        self._evaluate(session, feature_detector, handle_pl, eval_train_handle, is_training=True)

      except tf.errors.OutOfRangeError:
        break

    # Evaluate test set
    logging.info('Evaluating the test set...')
    eval_test_handle = session.run(test_iterator.string_handle())
    session.run(test_iterator.initializer)

    batch = 0
    while True:
      try:
        logging.info('Classifier Testing Batch #%s', batch)
        batch += 1
        self._evaluate(session, feature_detector, handle_pl, eval_test_handle, is_training=False)
      except tf.errors.OutOfRangeError:
        break

    return self._create_experiment(self.train_features, self.test_features, self.train_labels, self.test_labels,
                                   self.train_images, self.test_images)

  def _create_experiment(self, train_features, test_features, train_labels, test_labels, train_images=None,
                         test_images=None):
    """Combine features, labels and images from training and test sets, and create an Experiment instance."""
    classifier_features = np.concatenate([train_features, test_features], axis=0)
    classifier_labels = np.concatenate([train_labels, test_labels], axis=0)
    classifier_images = None
    if train_images is not None:
      classifier_images = np.concatenate([train_images, test_images], axis=0)
      logging.debug('classifier images: %s', str(classifier_images.shape))

    logging.debug('Features out size: %s, min/max: %s/%s',
                  str(classifier_features.shape),
                  str(classifier_features.min()),
                  str(classifier_features.max()))
    logging.debug('Labels out size: %s', str(classifier_labels.shape))

    # Export experiment for supervised analysis
    self._experiment = Experiment('classifier', classifier_features, classifier_labels, self._num_classes,
                                  classifier_images)
    return self._experiment

  def _evaluate(self, session, feature_detector, handle_pl, handle_value, is_training=True):
    """Evaluate features and labels to get concrete values."""

    batch_type = 'encoding'

    feed_dict = {
        handle_pl: handle_value
    }
    feature_detector.update_feed_dict(feed_dict, batch_type)

    # Build list of ops
    fetches = {}
    fetches['labels'] = self._labels # need to eval these I think
    if self._keep_images:
      fetches['images'] = self._images    # need to eval these I think
    feature_detector.add_fetches(fetches, batch_type)

    # Run the session
    fetched = session.run(fetches, feed_dict=feed_dict)

    # Store op results
    feature_detector.set_fetches(fetched, batch_type)

    # Get evaluated features & labels
    batch_features = feature_detector.get_features(batch_type)
    batch_labels = fetched['labels']

    batch_images = None
    if self._keep_images:
      batch_images = fetched['images']

    self.append(batch_features, batch_labels, is_training, batch_images)

  def _build_classifiers(self):
    """Build multiple classifiers depending on hparam search."""
    idx = 0
    classifiers = {}

    if self._hparams_search:
      for param_key, param_values in self._hparams_search.items():
        for param in param_values:
          search_param = str(param_key) + '=' + str(param)
          hparams = self._hparams + search_param

          classifier = Classifier(model=self._model,
                                  hparams_override=hparams,
                                  summary_dir=self._summary_dir,
                                  name='classifier-' + str(idx))
          classifiers[idx] = {'model': classifier, 'hparam': search_param}
          idx += 1
    else:
      classifier = Classifier(model=self._model,
                              hparams_override=self._hparams,
                              summary_dir=self._summary_dir)
      classifiers[idx] = {'model': classifier, 'hparam': ''}

    return classifiers

  def _classify(self, experiment, split_indices):
    """Perform classification using the Classifier component."""

    # Build multiple (optional) classifiers, depending on hparams_search
    classifiers = self._build_classifiers()

    # For each classifier, perform classification and return the results
    results = []
    for i, classifier in classifiers.items():
      logging.info('START: Classifier %i of %i: %s', i, len(classifiers) - 1, classifier['hparam'])

      result = classifier['model'].classify(
          experiment, split_indices, verbose=True,
          record_learning_curve=self._learning_curve,
          scale_features=self._scale_features)

      classifier['model'].classification_report(result)
      results.append(result)

      # Optional: Plot the learning curve
      if self._learning_curve:
        classifier.plot_learning_curve(
            filename='lc_'+ str(i) +'.png')

      logging.info('END: Classifier %i of %i: %s', i, len(classifiers) - 1, classifier['hparam'])

    return results

  def write_component_evaluation_summaries(self, feature_detector, step, writer):
    """
    Write the summaries that were created during evaluation phase for feature_detector
    i.e. Harness knows which batch_type was used
    """
    feature_detector.write_summaries(step, writer, batch_type='encoding')

  def export(self, output_dir='/tmp', filename=None):
    """Exports the experiment data to the specified output directory."""
    if self._experiment is None:
      logging.error('No experiment to export.')
      return

    logging.info('Exporting experiment')
    self._experiment.export_data(output_dir, filename)
