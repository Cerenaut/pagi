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


class HParamMulti:
  """
  Static helper methods for managing multiple hyperparam objects.
  The use case is for Components that have sub components.
  We cannot maintain state, because the flat tf hyperparam object is selectively overridden outside of the component.
  """

  @staticmethod
  def add(multi, source, component):
    """
    Prepend component namespace to hparams of source, and store in multi.

    :param multi: destination hparams object, set the values of this
    :param source: the source hparams object, whose values we'll use to set target
    :param component: the component that source corresponds to, used as namespace in target
    :return: the target hparam object
    """

    if multi is None or source is None or component is None:
      raise ValueError("One or more arguments was not defined.")

    for param in source.values():
      HParamMulti.set_param(multi, param, source.get(param), component)
    return multi

  @staticmethod
  def override(multi, target, component):
    """ Override hparams in target with corresponding value from multi """

    if multi is None or target is None or component is None:
      raise ValueError("One or more arguments was not defined.")

    for param in target.values():
      param_multi = component + '_' + param
      target.set_hparam(param, multi.get(param_multi))

    return target

  @staticmethod
  def set_param(multi, param, value, component):
    """ Set an individual hparam value in multi for a given component namespace, if it doesn't exist, add it """
    param_multi = component + '_' + param

    if multi.__contains__(param_multi):
      multi.set_hparam(param_multi, value)
    else:
      multi.add_hparam(param_multi, value)
