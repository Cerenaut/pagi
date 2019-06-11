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

"""Installation Instructions."""

import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'six >= 1.10.0',
    'numpy == 1.15.0',
    'scipy == 1.0.0'
    'scikit-learn == 0.19.1',
    'h5py == 2.8.0',
    'mlflow == 0.2.1',
    'matplotlib',
    'scikit-image'
]

class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(
   name='pagi',
   version='0.1.0',
   author='ProjectAGI',
   author_email='info@agi.io',
   packages=find_packages(),
   scripts=['bin/pagi'],
   url='https://github.com/ProjectAGI/pagi',
   license='Apache 2.0',
   description='A Tensorflow- based framework for building a selective memory system based on '
               'convolutional, hierarchical sparse autoencoder-like components.',
   long_description=long_description,
   long_description_content_type='text/markdown',
   install_requires=REQUIRED_PACKAGES,
   distclass=BinaryDistribution,
   cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    keywords='tensorflow memory machine learning',
)
