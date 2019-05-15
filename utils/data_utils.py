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

"""Generic data helper methods."""
import glob
import os

import tensorflow as tf

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
# From https://www.tensorflow.org/tutorials/load_data/tf_records
from utils.tf_utils import tf_print


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_subset(filename, dataset_shape, vals, labels, keep_fn):
  """
  Filter a dataset of images with the function 'keep_fn' and creates a TFRecordset
  Note:
    - designed for images
    - keep_fn compares to the whole batch
  """
  with tf.python_io.TFRecordWriter(filename) as writer:
    count = 0
    for idx, (image, label) in enumerate(zip(vals, labels)):
      if not keep_fn(idx, vals, labels):
        continue
      example = tf.train.Example(features=tf.train.Features(
          feature={
            'height': int64_feature(dataset_shape[2]),
            'width': int64_feature(dataset_shape[1]),
            'depth': int64_feature(dataset_shape[3]),
            'label': int64_feature(label),
            'image_raw': bytes_feature(image.tostring())}))
      writer.write(example.SerializeToString())
      count += 1

    print("Wrote {0} records".format(count))


def read_subset(filename, dataset_shape):
  """
  Complement of 'create_subset'. Reads in images in TFRecordset
  """

  def _parse_image_function(example_proto):

    # Create a dictionary describing the features.
    image_feature_description = {
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64),
      'label': tf.FixedLenFeature([], tf.int64),
      'image_raw': tf.FixedLenFeature([], tf.string),
    }

    # Parse the input tf.Example proto using the dictionary above.
    features = tf.parse_single_example(example_proto, image_feature_description)

    # Convert from a scalar string tensor (whose single string has
    # length image_pixel*image_pixel) to a float32 tensor with shape
    # [image_pixel, image_pixel, 1].
    width = dataset_shape[1]    # features['width']
    height = dataset_shape[2]   # features['height']
    depth = dataset_shape[3]
    images = tf.decode_raw(features['image_raw'], tf.float32)
    images = tf.reshape(images, [width, height, depth])
    images.set_shape([width, height, depth])

    labels = tf.to_int32(features['label'])

    return images, labels

  image_dataset = tf.data.TFRecordDataset(filename)
  dataset = image_dataset.map(_parse_image_function)

  return dataset


def generate_filenames(name, directory, data_file):
  dirpath = os.path.join(directory, name)
  filepath = os.path.join(dirpath, data_file)
  if not tf.gfile.Exists(dirpath):
    raise ValueError('Directory not found: ' + str(filepath))

  if data_file.startswith('sharded_'):
    sharded_file_format = data_file + '-*'
    tfrecords_list = glob.glob(os.path.join(dirpath, sharded_file_format))
    return tfrecords_list
  else:
    return filepath
