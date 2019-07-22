
import tensorflow as tf


class TbDebug(object):

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

