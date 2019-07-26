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

from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

install_requires = [
    'six >= 1.10.0',
    'numpy == 1.14.5',
    'scipy == 1.0.0',
    'scikit-learn == 0.19.1',
    'h5py == 2.8.0',
    'mlflow == 0.2.1',
    'matplotlib',
    'scikit-image',
    'wrapt',
    'click',
    'PyYAML <=3.13, >=3.10',
    'pre-commit'
]

setup_requires = []

extras_require = {
    'tf': ['tensorflow==1.10.0'],
    'tf_gpu': ['tensorflow-gpu == 1.10.0'],
    'tf_prob': ['tensorflow-probability==0.3.0']
}

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
    include_package_data=True,
    url='https://github.com/ProjectAGI/pagi',
    license='Apache 2.0',
    description='A Tensorflow-based framework for building a selective memory system based on '
                'convolutional, hierarchical sparse autoencoder-like components.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    keywords='tensorflow memory machine learning',
    entry_points={
        'console_scripts': [
            'pagi = pagi.scripts.cli:cli',
        ],
    },
)
