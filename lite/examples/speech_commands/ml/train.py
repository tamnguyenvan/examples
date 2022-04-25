# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow.compat.v1 as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from callbacks import ConfusionMatrixCallback
from model import speech_model, prepare_model_settings
from generator import AudioProcessor, prepare_words_list
from classes import get_classes
from utils import data_gen
import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.subplot(121)
    plt.plot(hist.epoch, hist.history['loss'], label='training')
    plt.plot(hist.epoch, hist.history['val_loss'], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.plot(hist.epoch, hist.history['categorical_accuracy'], label='training')
    plt.plot(hist.epoch, hist.history['val_categorical_accuracy'], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('result.png')


parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '-sample_rate',
    action='store',
    dest='sample_rate',
    type=int,
    default=16000,
    help='Sample rate of audio')
parser.add_argument(
  '-epochs',
  type=int,
  default=10,
  help='Number of epochs'
)
parser.add_argument(
    '-batch_size',
    action='store',
    dest='batch_size',
    type=int,
    default=32,
    help='Size of the training batch')
parser.add_argument(
    '-output_representation',
    action='store',
    dest='output_representation',
    type=str,
    default='raw',
    help='raw, spec, mfcc or mfcc_and_raw')
parser.add_argument(
    '-data_dirs',
    '--list',
    dest='data_dirs',
    nargs='+',
    required=True,
    help='<Required> The list of data directories. e.g., data/train')

parser.add_argument(
    '--classes',
    type=str,
    default='angry,bored',
    help='Classes would be trained')
parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=1000,
    help='Clip duration in ms')

args = parser.parse_args()
parser.print_help()
print('input args: ', args)

if __name__ == '__main__':
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  K.set_session(sess)
  data_dirs = args.data_dirs
  output_representation = args.output_representation
  sample_rate = args.sample_rate
  batch_size = args.batch_size
  epochs = args.epochs
#   classes = get_classes(wanted_only=True)
  classes = [x.strip() for x in args.classes.split(',')]
  model_settings = prepare_model_settings(
      label_count=len(prepare_words_list(classes)),
      sample_rate=sample_rate,
      clip_duration_ms=args.clip_duration_ms,
      window_size_ms=30.0,
      window_stride_ms=10.0,
      dct_coefficient_count=80,
      num_log_mel_features=60,
      output_representation=output_representation)

  print(model_settings)

  ap = AudioProcessor(
      data_dirs=data_dirs,
      wanted_words=classes,
      silence_percentage=13.0,
      unknown_percentage=60.0,
      validation_percentage=10.0,
      testing_percentage=10.0,
      model_settings=model_settings,
      output_representation=output_representation)
  train_gen = data_gen(ap, sess, batch_size=batch_size, mode='training')
  val_gen = data_gen(ap, sess, batch_size=batch_size, mode='validation')
  test_gen = data_gen(ap, sess, batch_size=batch_size, mode='testing')

  model = speech_model(
      'conv_1d_time_stacked',
      model_settings['fingerprint_size']
      if output_representation != 'raw' else model_settings['desired_samples'],
      # noqa
      num_classes=model_settings['label_count'],
      **model_settings)

  # embed()
  suffix = '-'.join(classes)
  checkpoints_path = os.path.join('checkpoints', f'conv_1d_time_stacked_model_{suffix}')
  if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

  callbacks = [
      # ConfusionMatrixCallback(
      #     val_gen,
      #     ap.set_size('validation') // batch_size,
      #     wanted_words=prepare_words_list(get_classes(wanted_only=True)),
      #     all_words=prepare_words_list(classes),
      #     label2int=ap.word_to_index),
      ReduceLROnPlateau(
          monitor='val_categorical_accuracy',
          mode='max',
          factor=0.5,
          patience=4,
          verbose=1,
          min_lr=1e-5),
      TensorBoard(log_dir='logs'),
      ModelCheckpoint(
          os.path.join(checkpoints_path,
                       'ep-{epoch:03d}-vl-{val_loss:.4f}.hdf5'),
          save_best_only=True,
          monitor='val_categorical_accuracy',
          mode='max')
  ]
  print('=' * 20)
  print('Training size: ', ap.set_size('training'))
  print('Validation size: ', ap.set_size('validation'))
  print(ap.set_size('validation') // batch_size)
  hist = model.fit(
      train_gen,
      steps_per_epoch=ap.set_size('training') // batch_size,
      epochs=epochs,
      validation_data=val_gen,
      validation_steps=ap.set_size('validation') // batch_size,
      verbose=1,
      callbacks=callbacks)
  plot_hist(hist)
  # eval_res = model.evaluate_generator(val_gen,
  #                                     ap.set_size('validation') // batch_size)
                                    
  # print(eval_res)
  test_loss, test_acc = model.evaluate_generator(
    test_gen,
    ap.set_size('testing') // batch_size
  )
  print(f'Testing loss: {test_loss:.4f} Testing accuracy: {test_acc:.4f}')

  with open(f'{suffix}_l{test_loss}_acc{test_acc}.txt', 'wt') as f:
    f.write('.')
