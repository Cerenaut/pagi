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

"""Generic helper methods."""

import os
import sys
import math
import json
import random
import logging
import inspect
import datetime
import importlib

import numpy as np
import tensorflow as tf

from absl import logging


def get_summary_dir():
  now = datetime.datetime.now()
  summary_dir = './run/summaries_' + now.strftime("%Y%m%d-%H%M%S") + '/'
  return summary_dir

def get_logging():
  level = logging.getLogger().getEffectiveLevel()
  return level

def set_logging(level):
  logging._warn_preinit_stderr = 0  # pylint: disable=protected-access
  log_level = parse_logging_level(level)

  logging.set_verbosity(log_level)

  logging.debug("Python Version: %s", sys.version)


def set_seed(seed):
  """Set the seed for numpy and tf PRNGs."""
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)


def parse_logging_level(level):
  """
  Map the specified level to the numerical value level for the logger
  :param level: Logging level from command argument
  """
  try:
    level = level.lower()
  except AttributeError:
    level = ""

  return {
      'debug': logging.DEBUG,
      'info': logging.INFO,
      'warning': logging.WARNING,
      'error': logging.ERROR,
      'fatal': logging.FATAL
  }.get(level, logging.WARNING)


def get_sqrt_shape(area):
  """
  Returns a square shape, assuming the area provided can be square.
  """
  height = int(math.sqrt(float(area)))
  width = height
  return width, height


def get_module_class_ref(module_name, module_filepath=None):
  """Imports a module by name (or filepath) and gets a reference to first class in the module."""
  module_class = None

  def resolve_filename(dir_path, module_name):
    filename = os.path.join(dir_path, *module_name.split('.'))
    if os.path.isdir(filename):
      filename = os.path.join(filename, '__init__.py')
    else:
      filename += '.py'
    return filename

  try:
    if not module_name.startswith('pagi.'):
      module_filepath = resolve_filename(os.getcwd(), module_name)

    if module_filepath:
      spec = importlib.util.spec_from_file_location(module_name, module_filepath)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
    else:
      module = importlib.import_module(module_name)

    # Identify all classes in the module excluding class imports
    module_classes = inspect.getmembers(module,
                                        lambda member: inspect.isclass(member) and member.__module__ == module.__name__)

    # Get the first class object
    _, module_class = module_classes[0]
  except:
    raise NotImplementedError('Not implemented: ' + module_name)

  return module_class

def class_filter(dataset, classes, is_superclass=False, proportion=1.0):
  """
  Handles filtering of (super)classes for use with tf.Dataset.

  Arguments:
    dataset: An instance of the Dataset class.
    classes: A list of classes (or superclasses).
    is_superclass: A flag indicate whether or not the "classes" param consists of superclasses.
    proportion: A float indicating the proportion of classes to retrieve.
  """
  output_classes = []

  try:
    if classes and is_superclass:
      output_classes = dataset.get_classes_by_superclass(classes, proportion)
    elif classes and not is_superclass:
      output_classes = [int(x) for x in classes]
  except:
    raise ValueError('Failed to filter classes.')

  return output_classes

def load_exp_config(flags):
  """Load experiment config from JSON, and override tf.FLAGS."""
  exp_config = None

  if flags.experiment_def:
    with open(flags.experiment_def) as config_file:
      exp_config = json.load(config_file)

    # Override flags from file
    if 'experiment-options' in exp_config:
      for key, value in exp_config['experiment-options'].items():
        if not key.endswith('_sweep'):  # Don't override sweep parameters
          flags[key].value = value

  return exp_config
