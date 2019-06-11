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

"""Experiment framework for training and evaluating COMPONENTS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import json

import click
import mlflow
import tensorflow as tf

from pagi.utils import logger_utils
from pagi.utils import generic_utils as util
from pagi.utils.tb_debug import TbDebug



def run_experiment(flags, exp_config):
  """Setup and execute an experiment workflow with specified options."""
  util.set_logging(flags['logging'])

  TbDebug.TB_DEBUG = flags['tb_debug']

  # Get the component's default HParams, then override
  # -------------------------------------------------------------------------
  component_hparams_override = {}

  # Override that if defined using flag
  if flags['hparams_override']:
    if isinstance(flags['hparams_override'], dict):
      component_hparams_override = flags['hparams_override']
    else:
      # Unstringy the string formatted dict
      component_hparams_override = ast.literal_eval(flags['hparams_override'])

  # Override hparams for this sweep/run
  if flags['hparams_sweep']:
    # Unstringy the string formatted dict
    hparams_sweep = ast.literal_eval(flags['hparams_sweep'])

    # Selectively override component hparams
    component_hparams_override.update(hparams_sweep)

  # Export settings
  # -------------------------------------------------------------------------
  export_opts = {
      'export_filters': True,
      'export_checkpoint': True,
      'interval_batches': flags['batches']
  }

  # Classifier settings
  # -------------------------------------------------------------------------
  classifier_opts = {
      'model': 'logistic',  # Options: logistic, svm
      'unit_range': False,  # Set to True if using SVM
      'interval_batches': flags['batches'],
      'hparams': {
          'logistic': {
              'C': [0.01, 0.1, 1.0, 10.0]  # Regularization
          },
          'svm': {
              'C': [1.0, 10.0, 100.0]  # Regularization
          }
      }
  }

  # Checkpoint Options
  # -------------------------------------------------------------------------
  checkpoint_opts = {
      'checkpoint_path': flags['checkpoint'],
      'checkpoint_load_scope': flags['checkpoint_load_scope'],
      'checkpoint_frozen_scope': flags['checkpoint_frozen_scope']
  }

  # OPTIONAL: Override options from an experiment definition file
  # -------------------------------------------------------------------------
  workflow_opts_override = {}

  if exp_config:
    if 'export-options' in exp_config:
      export_opts.update(exp_config['export-options'])
    if 'workflow-options' in exp_config:
      workflow_opts_override.update(exp_config['workflow-options'])
    if 'classifier-options' in exp_config:
      classifier_opts.update(exp_config['classifier-options'])
    if 'checkpoint-options' in exp_config:
      checkpoint_opts.update(exp_config['checkpoint-options'])

  # Override workflow options for this sweep/run
  if flags['workflow_opts_sweep']:
    # Unstringy the string formatted dict
    workflow_opts_sweep = ast.literal_eval(flags['workflow_opts_sweep'])

    # Selectively override component hparams
    workflow_opts_override.update(workflow_opts_sweep)

  # Training with Tensorflow
  # -------------------------------------------------------------------------
  with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    if TbDebug.TB_DEBUG:
      from tensorflow.python import debug as tf_debug
      session = tf_debug.TensorBoardDebugWrapperSession(session, 'localhost:6064')
      print("Use the following command to run Tensorboard Debugger:\n'tensorboard --logdir=./ --debugger_port 6064'")

    util.set_seed(flags['seed'])

    # Load relevant dataset, workflow and component modules
    dataset_class = util.get_module_class_ref(flags['dataset'])
    workflow_class = util.get_module_class_ref(flags['workflow'])
    component_class = util.get_module_class_ref(flags['component'])

    # Override workflow options
    workflow_opts = workflow_class.default_opts()
    workflow_opts.override_from_dict(workflow_opts_override)

    # Log experiment settings
    print('Dataset:', flags['dataset'])
    print('Workflow:', flags['workflow'])
    print('Component:', flags['component'], '\n')

    print('Export Options:', json.dumps(export_opts, indent=4))
    print('Workflow Options:', json.dumps(workflow_opts.values(), indent=4))
    print('Classifier Options:', json.dumps(classifier_opts, indent=4))
    print('Checkpoint Options:', json.dumps(checkpoint_opts, indent=4), '\n')

    # Setup Experiment Workflow
    # -------------------------------------------------------------------------
    workflow = workflow_class(session, dataset_class, flags['dataset_location'], component_class,
                              component_hparams_override, classifier_opts, export_opts, opts=workflow_opts,
                              summarize=flags['summarize'], seed=flags['seed'], summary_dir=flags['summary_dir'],
                              checkpoint_opts=checkpoint_opts)

    # Start experiment to train the model and evaluating every N batches
    # -------------------------------------------------------------------------
    workflow.run(flags['batches'], evaluate=workflow_opts.evaluate, train=workflow_opts.train)

    session.close()


@click.command()
@click.option('--dataset', default='pagi.datasets.mnist_dataset',
              help='The dataset to use for the experiment.'
                   'Refer to the ./datasets/ directory for supported datasets.')
@click.option('--workflow', default='pagi.workflows.workflow',
              help='The workflow to use for the experiment.'
                   'Refer to the ./workflows/ directory for workflow options.')
@click.option('--component', default='pagi.components.sparse_conv_autoencoder_component',
              help='The component to use for the experiment.'
                   'Refer to the ./components/ directory for supported components.')

@click.option('--dataset_location', default='data',
              help='The location of the dataset. Note that for some sets '
                   'such as mnist, it is downloaded for you, so you can '
                   'leave this blank')

@click.option('--hparams_override', default=None,
              help='The hyperparameters to override for this experiment.')
@click.option('--hparams_sweep', default=None,
              help='Jenkins ONLY - The hyperparameters to override for this run/sweep.')
@click.option('--workflow_opts_sweep', default=None,
              help='Jenkins ONLY - The workflow options to override for this run/sweep.')

@click.option('--logging', type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']), default='info',
              help='Verbosity level for logging.')
@click.option('--checkpoint', default=None,
              help='A saved checkpoint for evaluation or further training.')
@click.option('--checkpoint_load_scope', default=None,
              help='Choose which variable/name scopes to load using a comma-separated list.')
@click.option('--checkpoint_frozen_scope', default=None,
              help='Choose which variable/name scopes to freeze using a comma-separated list.')

@click.option('--summary_dir', default=None,
              help='Explicitly defines the experiment summary directory.')
@click.option('--experiment_def', default=None,
              help='Overrides experiment options from a JSON definition file.')

@click.option('--seed', default=42,
              help='Seed used to control randomness for reproducability.')
@click.option('--batches', default=10,
              help='Number of batches to train for.')
@click.option('--experiment_id', default=None,
              help='The experiment identifier generated by MLFlow.')

@click.option('--summarize/--no-summarize', default=True,
              help='Enable summaries during training.')
@click.option('--track/--no-track', default=False,
              help='Track experiment using mlflow.')
@click.option('--tb_debug/--no-tb_debug', default=False,
              help='Debug with tensorboard debugger.')

def main(dataset, workflow, component, dataset_location, hparams_override, hparams_sweep, workflow_opts_sweep,
         logging, checkpoint, checkpoint_load_scope, checkpoint_frozen_scope, summary_dir, experiment_def, seed,
         batches, experiment_id, summarize, track, tb_debug):
  """Entrypoint for the command-line tool."""

  flags = locals()

  # OPTIONAL: Override flags from an experiment definition file.
  # Flags set in the experiment definition file have precedent over any flags
  # set by the command line, except sweep-related parameters.
  # -------------------------------------------------------------------------
  exp_config = None
  if experiment_def:
    with open(experiment_def) as config_file:
      exp_config = json.load(config_file)

    # Override flags from file
    if 'experiment-options' in exp_config:
      for key, value in exp_config['experiment-options'].items():
        if not key.endswith('_sweep'):  # Don't override sweep parameters
          flags[key].value = value

  if track:
    with mlflow.start_run(experiment_id=experiment_id):
      logger_utils.log_param({'num_batches': batches})

      run_experiment(flags, exp_config)
  else:
    run_experiment(flags, exp_config)
