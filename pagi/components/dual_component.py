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

"""DualComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pagi.utils.dual import DualData
from pagi.components.component import Component


class DualComponent(Component):
  """
  A Component that uses a DualData object to manage on/off graph data.
  It also has a unique name() property.
  """

  def __init__(self, name=None):
    super().__init__()

    self._dual = DualData(name)
    # self._name = name  # Maybe discovered after instantiation time

  @property
  def name(self):
    return self._dual.get_root_name()

  @name.setter
  def name(self, name):
    self._dual.set_root_name(name)

  def get_dual(self):
    return self._dual

  def get_op(self, key):
    return self._dual.get_op(key)

  def get_shape(self, key):
    return self._dual.get(key).get_shape()

  def get_values(self, key):
    return self._dual.get_values(key)
