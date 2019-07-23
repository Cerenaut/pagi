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

"""TbDebug class."""

import tensorflow as tf


class TbDebug:
  """TensorBoard debugger."""

  TB_DEBUG = False

  def __init__(self):
    pass

  @staticmethod
  def tf_debug_monitor(tensor, name):
    if not TbDebug.TB_DEBUG:
      return tensor

    var = tf.get_variable(name=name, shape=tensor.get_shape(), trainable=False)
    tensor = tf.assign(var, tensor)
    return tensor
