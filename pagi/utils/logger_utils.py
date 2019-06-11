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

"""Logger methods."""

import mlflow
import tensorflow as tf


def validate_dict(input_dict):
  if not isinstance(input_dict, dict):
    raise ValueError('Expected a dict, but received %s.' % type(input_dict).__name__)


def log_param(hparams):
  """Log multiple hyperparameters from a dict."""
  if not tf.flags.FLAGS.track:
    return

  if isinstance(hparams, tf.contrib.training.HParams):
    hparams = hparams.values()

  validate_dict(hparams)

  for key, value in hparams.items():
    mlflow.log_param(key, value)


def log_metric(metrics):
  """Log multiple metrics from a dict."""
  if not tf.flags.FLAGS.track:
    return

  validate_dict(metrics)

  for key, value in metrics.items():
    mlflow.log_metric(key, value)
