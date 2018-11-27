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

"""Model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from sklearn.model_selection import learning_curve

class Model(object):
  """
  The base class for building a classifcation models and running inference
  on them.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, hparams, batch_size=None, num_classes=None,
               summary_dir=None, verbose=False):
    """
    Initializes the model parameters.

    Args:
        hparams: The hyperparameters for the model as
            tf.contrib.training.HParams.
        batch_size: An integer, the number of samples in a batch.
        num_classes: An integer, the number of classes.
        summary_dir: The output directory for the results.
        verbose: A boolean to enable verbose logging.
    """
    self._model = None
    self._hparams = hparams
    self._verbose = verbose
    self._batch_size = batch_size
    self._num_classes = num_classes
    self._summary_dir = summary_dir

  @abc.abstractstaticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    raise NotImplementedError('Not implemented')

  @abc.abstractmethod
  def train(self, features, labels, seed=None):
    """
    Setup the model with specified hyperparameters and train the model.

    Args:
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing the label for each sample.
        seed: An integer used to specify the randomness seed.
    """
    raise NotImplementedError('Not implemented')

  @abc.abstractmethod
  def evaluate(self, features, labels):
    """
    Evaluates the trained model using the specified features and labels.

    Args:
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing the label for each sample.
    Returns:
        accuracy: The accuracy score of the model.
        predictions: The labels predicted by the model for each sample.
    """
    raise NotImplementedError('Not implemented')

  def learning_curve(self, features, labels):
    """Simple wrapper around sklearn's learning curve module"""
    return learning_curve(self._model, features, labels)
