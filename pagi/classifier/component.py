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

"""Component interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Component(object):
  """This is the basic interface for a graph-component that can be executed as a learning algorithm."""

  @staticmethod
  def default_hparams():
    return tf.contrib.training.HParams()

  def reset(self):
    pass

  def get_batch_type(self):
    pass

  def update_feed_dict(self, feed_dict, batch_type='training'):
    """Add items to the feed dict to run a batch."""
    pass

  def add_fetches(self, fetches, batch_type='training'):
    """Add graph ops to the fetches dict so they are evaluated."""
    pass

  def set_fetches(self, fetched, batch_type='training'):
    """Store results of graph ops in the fetched dict so they are available as needed."""
    pass

  def build_summaries(self, batch_types=None, scope=None):
    """Build any summaries needed to debug the module."""
    pass

  def write_summaries(self, step, writer, batch_type='training'):
    """Write any summaries needed to debug the module."""
    pass
