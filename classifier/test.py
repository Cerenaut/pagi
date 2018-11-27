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

"""Classification test case using the Iris dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.datasets import load_iris

from experiment import Experiment
from classifier import Classifier

features_shape = (150, 4)
labels_shape = (150, 3)


def main():
  # Load dataset
  iris = load_iris()
  features = iris.data  # pylint: disable=E1101
  labels = iris.target
  classes = list(iris.target_names)  # pylint: disable=E1101

  # Setup Experiment
  experiment = Experiment('iris', num_classes=len(classes),
                          features=features, labels=labels)

  # Exporting
  # saves to ./classify/output/output_iris.h5
  experiment.export_data(output_dir='./')

  features, labels, _ = experiment.get_data()
  assert features.shape == features_shape
  assert labels.shape == labels.shape

  # Importing
  experiment = None
  experiment = Experiment('iris', num_classes=len(classes))
  experiment.import_data('./classify/output/output_iris.h5')

  features, labels, _ = experiment.get_data()

  # Validate data imported OK
  assert features.shape == features_shape
  assert labels.shape == labels.shape

  split_indices = {
      'train': (0, 100),
      'test': (100, 150)
  }

  # Setup Classifier with default hparams
  classifier = Classifier('svm', summary_dir='./classify/')
  results = classifier.classify(
      experiment, split_indices, verbose=True, seed=42, shuffle=True,
      record_learning_curve=True)

  classifier.classification_report(results)
  classifier.plot_learning_curve()


if __name__ == '__main__':
  main()
