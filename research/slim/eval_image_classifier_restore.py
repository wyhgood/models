#coding=utf-8
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
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat


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
  sess = tf.Session()
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()
    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)
    
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    img = Image.open("test.jpg")
    img = img.resize((eval_image_size, eval_image_size))
    img_np = numpy.array(img).reshape(( eval_image_size, eval_image_size, 3))
    print(img_np.shape)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    images_placeholder = tf.placeholder(tf.float32, shape=(eval_image_size, eval_image_size, 3))
    
    image = image_preprocessing_fn(images_placeholder, eval_image_size, eval_image_size)
    image_input = tf.reshape(image, [1, eval_image_size, eval_image_size, 3])
    '''
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    print(images.shape)
          
    images_placeholder = tf.placeholder(tf.float32, shape=(1, eval_image_size, eval_image_size, 3))
    '''
    

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(image_input)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)


    checkpoint_path = "v3_model/model.ckpt-91974"

    session_creator = monitored_session.ChiefSessionCreator(
        checkpoint_filename_with_path=checkpoint_path)
    with monitored_session.MonitoredSession(
        session_creator=session_creator) as session:
          '''
          starttime = datetime.datetime.now()
          print(starttime)
          list = session.run(predictions, feed_dict={images_placeholder:img_np})
          endtime = datetime.datetime.now()
          print(endtime)
          print(list)
          
          boxes_output_tensor_info = utils.build_tensor_info(boxes)

          scores_output_tensor_info = utils.build_tensor_info(scores)

          classes_output_tensor_info = utils.build_tensor_info(classes)

          num_detections_output_tensor_info = utils.build_tensor_info(num_detections)
 
          '''
          #session.graph._unsafe_unfinalize() 
          output_path = "/tmp/output/1"

          builder = saved_model_builder.SavedModelBuilder(output_path) 
          predictions_output_tensor_info = utils.build_tensor_info(predictions)
          image_inputs_tensor_info = utils.build_tensor_info(images_placeholder)
          print('0000000000000000000')
        
          classification_signature = signature_def_utils.build_signature_def(
              inputs={
                  'constant_input_image': image_inputs_tensor_info
              },
              outputs={
                  'constant_output_predictions':
                      predictions_output_tensor_info
              },
              method_name=signature_constants.CLASSIFY_METHOD_NAME)
     
          print('11111111111111111111')   
          legacy_init_op = tf.group(
              tf.initialize_all_tables(), name='legacy_init_op')   #table是一个字符串的映射表
          builder.add_meta_graph_and_variables(
              session, 
              signature_def_map={
                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                      classification_signature,
              },
              legacy_init_op=legacy_init_op)          #将sess和变量保存至SavedModel
          print('8888888888888888888')
          #session.graph.finalize()
          builder.save() 
          
          #模型保存在SavedModel初始化的路径中
          print('Successfully exported model to ' + output_path)       
  
if __name__ == '__main__':
  tf.app.run()
