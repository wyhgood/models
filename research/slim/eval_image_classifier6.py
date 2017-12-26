# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import tensorflow as tf
from PIL import Image
import numpy
import datetime
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(1001 - FLAGS.labels_offset),
        is_training=False)
    
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    image_map = {}
    for root, dirs, files in os.walk('./test_image/'):
        for f in files:
            path = root+f
            print(path)
            img = Image.open(path)
            img_np = numpy.asarray(img).astype('float')
            image_map[path] = img_np/255
            print(img_np/255)

    preprocessing_name = 'inception_v3'
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    images_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3))
        
    image = image_preprocessing_fn(images_placeholder, eval_image_size, eval_image_size)
    image_input = tf.reshape(image, [1, eval_image_size, eval_image_size, 3])
    

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(image_input)


    predictions = tf.argmax(logits, 1)
    checkpoint_path = "check_point_threelable/model.ckpt-21764"
    #checkpoint_path = "check_point/model.ckpt-91974"
    #checkpoint_path = "check_point/model.ckpt-201579"
    #checkpoint_path = "check_point/model.ckpt-122647"
    #saver = tf.train.import_meta_graph('check_point/model.ckpt.meta')
    saver = tf.train.Saver()
    with tf.Session() as sess:
          saver.restore(sess, checkpoint_path)
          for key in image_map:
              print(key)
              
              list = sess.run(predictions, feed_dict={images_placeholder:image_map[key]})
              
              print(list)
if __name__ == '__main__':
  tf.app.run()
