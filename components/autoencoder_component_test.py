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

"""AutoencoderComponentTest class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from components.autoencoder_component import AutoencoderComponent


class AutoencoderComponentTest(tf.test.TestCase):
  """Unit tests for Autoencoder Component class."""

  def setUp(self):
    # use this hparam to override the Component's default hparams in the tests
    self.hparams_override = {
        "learning_rate": 0.05,
        "batch_size": 1,
        "loss_type": 'mse',
        "filters": 64}

    self.toy_input = np.reshape(np.arange(28 * 28),
                                (self.hparams_override["batch_size"], 784))
    self.toy_input_shape = [self.hparams_override["batch_size"], 28, 28, 1]

  def test_training_basic(self):
    """Tests the training operation and intended variables are trainable."""
    with tf.Graph().as_default():
      toy_input_tensor = tf.constant(self.toy_input, dtype=tf.float32)

      hparams = AutoencoderComponent.default_hparams()
      hparams.override_from_dict(self.hparams_override)

      component = AutoencoderComponent()
      component.build(toy_input_tensor, self.toy_input_shape, hparams)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

        batch_type = 'training'

        fetches = {}
        component.add_fetches(fetches, batch_type)

        feed_dict = {}
        component.update_feed_dict(feed_dict, batch_type)

        fetched = sess.run(fetches, feed_dict)

        component.set_fetches(fetched, batch_type)
        loss = component._dual.get_op('loss')

        self.assertEqual(loss.get_shape(), ())
        self.assertEqual(len(trainable_vars), 3)


if __name__ == '__main__':
  tf.test.main()
