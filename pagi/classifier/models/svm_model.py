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

"""SvmModel class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sklearn import svm
from sklearn import metrics

from pagi.classifier.models import model


class SvmModel(model.Model):
  """
  An Svm model that utilises the sklearn.svm module for training and
  inference. The default usage of Svm is non-linear; however, a linear
  kernel can be specified as a hyperparameter.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        C=100.0,
        eps=0.001,
        gamma=0.1,
        kernel='rbf',
        shrinking=True, # Whether or not to use shrinking heuristics.
        max_iter=-1 # Defaults to no limit (-1), override to set hard limit
    )

  def train(self, features, labels, seed=None):
    """
    Setup the Svm model with specified hyperparameters,
    and train the model.

    Args:
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing the label for each sample.
        seed: An integer used to specify the randomness seed.
    """
    self._model = svm.SVC(kernel=self._hparams.kernel, C=self._hparams.C,
                          gamma=self._hparams.gamma, tol=self._hparams.eps,
                          verbose=self._verbose, random_state=seed,
                          shrinking=self._hparams.shrinking,
                          max_iter=self._hparams.max_iter)

    self._model = self._model.fit(features, labels)

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
    predictions = self._model.predict(features)
    accuracy = metrics.accuracy_score(labels, predictions)

    return accuracy, predictions
