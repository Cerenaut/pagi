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

"""LogisticRegressionModel class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from pagi.classifier.models import model


class LogisticRegressionModel(model.Model):
  """
  A Logistic Regression model that utilises the
  sklearn.linear_model.LogisticRegression module for training and inference.
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        C=1.0,
        eps=0.001,
        bias=True,
        penalty='l2',
        multi_class='ovr',
        solver='liblinear'
    )

  def train(self, features, labels, seed=None):
    """
    Setup the Logistic Regression model with specified hyperparameters,
    and train the model.

    Args:
        features: A numpy n-dimensional array containing the features in
            the shape of samples x features.
        labels: A numpy array containing the label for each sample.
        seed: An integer used to specify the randomness seed.
    """
    self._model = LogisticRegression(solver=self._hparams.solver,
                                     C=self._hparams.C,
                                     tol=self._hparams.eps,
                                     fit_intercept=self._hparams.bias,
                                     multi_class=self._hparams.multi_class,
                                     penalty=self._hparams.penalty,
                                     verbose=self._verbose,
                                     random_state=seed)

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
