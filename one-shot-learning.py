import argparse
import math
import os
import random
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid, tanh

from AssociativeLSTM import *
from dnc import *
from OmniGlotDatasetReader import *


Config = namedtuple('Config',
                    ['batch_size', 'cell_size', 'cell_type', 'num_classes',
                     'elements_per_class', 'sequence_len', 'learning_rate',
                     'train_epochs', 'test_epochs', 'train_seqs', 'test_seqs',
                     'input_keys', 'output_keys', 'num_copies',
                     'memory_locations', 'location_size', 'read_heads',
                     'write_heads'])

class OneShotLearner:

  def __init__(self, data_reader, config):

    self.reader = data_reader
    # total_testing_sequences = batch_size * test_epochs
    # total_training_sequences = batch_size * train_epochs

    self.batch_size = config.batch_size
    self.cell_size = config.cell_size
    self.cell_type = config.cell_type
    self.num_classes = config.num_classes
    self.elements_per_class = config.elements_per_class
    self.sequence_len = config.sequence_len
    self.learning_rate = config.learning_rate
    self.total_sequence_len = self.num_classes * self.elements_per_class
    self.train_epochs = config.train_epochs
    self.test_epochs = config.test_epochs

    self.out_size = self.num_classes

    #ALSTM related config
    if self.cell_type == 'alstm':
      self.input_keys = config.input_keys
      self.output_keys = config.output_keys
      self.num_copies = config.num_copies
    elif self.cell_type == 'dnc':
      self.memory_locations = config.memory_locations
      self.location_size = config.location_size
      self.read_heads = config.read_heads
      self.write_heads = config.write_heads
    else:
      self.input_keys = 1
      self.output_keys = 1
      self.num_copies = 1

    self._create_graph()

  def _create_placeholders(self):
    # a batch is generated to know the dimensionality of each element in the
    # sequence
    batch, _ = self.reader.generate_batch(
        batch_size = 1, num_classes = self.num_classes,
        sequence_length = self.num_classes * self.elements_per_class)

    with tf.name_scope('placeholders'):
      self.data = tf.placeholder(tf.float32,
                                 [self.batch_size, self.total_sequence_len,
                                  batch.shape[2]])
      self.target = tf.placeholder(tf.float32,
                                   [self.batch_size, self.total_sequence_len,
                                     self.num_classes])

    with tf.name_scope('global_step'):
      self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                     name='global_step')

    del batch

  def _create_cell(self):
  # UnitaryRNN, Peepholes, depth, bla bla
    with tf.name_scope('cell'):
      if self.cell_type == 'rnn':
        self.cell = tf.nn.rnn_cell.RNNCell(self.cell_size)
      elif self.cell_type == 'lstm':
        self.cell = tf.nn.rnn_cell.LSTMCell(self.cell_size)
      elif self.cell_type == 'gru':
        self.cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
      elif self.cell_type == 'alstm':
        self.cell = AssociativeLSTMCell(self.cell_size, self.num_copies,
            input_keys = self.input_keys, output_keys = self.output_keys,
            num_proj = self.cell_size * self.output_keys)
      elif self.cell_type == 'dnc':
        self.cell = DNCCell(self.memory_locations, self.location_size,
            read_heads = self.read_heads, write_heads = self.write_heads)
      else:
        raise ValueError

  def _create_output_weights(self):
    with tf.variable_scope('output_layer'):
      if self.cell_type == 'alstm':
        self.W = tf.get_variable('W',
                    [self.cell_size * self.output_keys, self.out_size],
                    initializer = tf.contrib.layers.xavier_initializer())
      elif self.cell_type == 'dnc':
        self.W = tf.get_variable('W',
                    [self.location_size, self.out_size],
                    initializer = tf.contrib.layers.xavier_initializer())
      else:
        self.W = tf.get_variable('W',
                    [self.cell_size, self.out_size],
                    initializer = tf.contrib.layers.xavier_initializer())

      self.b = tf.get_variable('b', [self.out_size],
                  initializer = tf.zeros_initializer)

  def _compute_output(self):
    # cell_output: bs x seq_len x cell_size
    # cell_state: bs x cell_size
    with tf.name_scope('cell_output'):
      (cell_output, cell_state) = tf.nn.dynamic_rnn(self.cell, self.data,
                                                    dtype = tf.float32)

    max_time = self.total_sequence_len

    with tf.name_scope('model_output'):
      if self.cell_type == 'alstm':
        output = tf.reshape(cell_output, [-1, self.cell_size * self.output_keys])
      elif self.cell_type == 'dnc':
        output = tf.reshape(cell_output, [-1, self.location_size])
      else:
        output = tf.reshape(cell_output, [-1, self.cell_size])
      prediction = tf.matmul(output, self.W) + self.b
      self.prediction = tf.reshape(prediction, [-1, max_time, self.out_size])

  def _compute_loss(self):
    with tf.name_scope('loss'):
      loss = tf.nn.softmax_cross_entropy_with_logits(
                      logits = self.prediction, labels = self.target)
      self.loss = tf.reduce_sum(loss)

  def _create_optimizer(self):
    with tf.name_scope('optimizer'):
      optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
      gvs = optimizer.compute_gradients(self.loss)
      capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs,
                                                global_step = self.global_step)

  def _compute_evaluation_metrics(self):
    #TODO: Compute accuracy using tf.metrics
    #TODO: Compute the first/second/fifth/tenth time accuracies
    # Computing Accuracy
    with tf.name_scope('evaluation'):
      pred_labels = tf.argmax(self.prediction[:, -self.sequence_len:, :], 2)
      true_labels = tf.argmax(self.target[:, -self.sequence_len:, :], 2)

      equality = tf.equal(pred_labels, true_labels)
      self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

  def _create_summaries(self):
    with tf.name_scope('train_summaries'):
      train_loss_summary = tf.summary.scalar("loss", self.loss)
      train_accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
      self.summary_op = tf.summary.merge([train_loss_summary,
                                          train_accuracy_summary])

    with tf.name_scope('val_summaries'):
      val_loss_summary = tf.summary.scalar("loss", self.loss)
      val_accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
      self.val_summary_op = tf.summary.merge([val_loss_summary,
                                              val_accuracy_summary])

  def _create_graph(self):
    self._create_placeholders()
    self._create_cell()
    self._create_output_weights()
    self._compute_output()
    self._compute_loss()
    self._create_optimizer()
    self._compute_evaluation_metrics()
    self._create_summaries()

  def train(self):

    X_test, y_test = omni_glot_reader.generate_batch(
        batch_size = self.batch_size, num_classes = self.num_classes,
        sequence_length = self.num_classes * self.elements_per_class,
        training = False)


    # Add an op to initialize the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:

      sess.run(init)

      # Load saved model
      ckpt = tf.train.get_checkpoint_state(
                  os.path.dirname('checkpoints/'
                                  + self.cell_type
                                  + '/checkpoint'))
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

      writer = tf.summary.FileWriter('model/' + self.cell_type, sess.graph)

      for e in range(self.train_epochs):
        for batch_index  in range(self.train_epochs):
          if batch_index > 0 or e > 0:
            tf.get_variable_scope().reuse_variables()

          current_x_batch, current_y_batch = omni_glot_reader.generate_batch(
              batch_size = self.batch_size, num_classes = self.num_classes,
              sequence_length = self.num_classes * self.elements_per_class)


          _, l, train_acc, summary, i = sess.run([self.train_op,
                                       self.loss, self.accuracy,
                                       self.summary_op, self.global_step],
                                  feed_dict = {self.data: current_x_batch,
                                               self.target: current_y_batch})
          writer.add_summary(summary, global_step=i)
          print('batch %d loss = %1.5f' % (batch_index , l))


        test_loss, acc, summary = sess.run([self.loss, self.accuracy,
                                            self.val_summary_op],
                              feed_dict = {self.data: X_test,
                                           self.target: y_test})
        writer.add_summary(summary, global_step=i)
        print('train loss: ', l, ' train acc: ', train_acc,
              ' test loss: ', test_loss, '  test acc: ', acc)
        save_path = saver.save(sess, 'checkpoints/'
                               + self.cell_type
                               + '/',
                               i)

if __name__ == '__main__':
  np.random.seed(seed = 1234)

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type = int, default = 16)
  parser.add_argument('--cell-size', type = int, default = 32)
  parser.add_argument('--cell-type', default = 'alstm')
  parser.add_argument('--num-classes', type = int, default = 5)
  parser.add_argument('--elements-per-class', type = int, default = 10)
  parser.add_argument('--sequence-len', type = int, default = 5)
  parser.add_argument('--lr', type = float, default = 5e-4)
  parser.add_argument('--train-epochs', type = int, default = 10)
  parser.add_argument('--test-epochs', type = int, default = 1)
  parser.add_argument('--train-seqs', type = int, default = 16 * 100)
  parser.add_argument('--test-seqs', type = int, default = 16 * 1)

  # ALSTM related
  parser.add_argument('--input-keys', type = int, default = 4)
  parser.add_argument('--output-keys', type = int, default = 16)
  parser.add_argument('--num-copies', type = int, default = 16)

  # DNC related
  parser.add_argument('--memory-locations', type = int, default = 16)
  parser.add_argument('--location-size', type = int, default = 20)
  parser.add_argument('--read-heads', type = int, default = 2)
  parser.add_argument('--write-heads', type = int, default = 1)

  args = parser.parse_args()

  config = Config(batch_size = args.batch_size, cell_size = args.cell_size,
                  cell_type = args.cell_type, num_classes = args.num_classes,
                  elements_per_class = args.elements_per_class,
                  sequence_len = args.sequence_len, learning_rate = args.lr,
                  train_epochs = args.train_epochs,
                  test_epochs = args.test_epochs, train_seqs = args.train_seqs,
                  test_seqs = args.test_seqs, input_keys = args.input_keys,
                  output_keys = args.output_keys, num_copies = args.num_copies,
                  memory_locations = args.memory_locations,
                  location_size = args.location_size,
                  read_heads = args.read_heads, write_heads = args.write_heads)


  train_directories = ['./Data/images_background_small1',
                        './Data/images_background_small2']
  evaluation_directories = ['./Data/images_evaluation']

  omni_glot_reader = OmniGlotDatasetReader(train_directories,
                                           evaluation_directories)

  omniglot_oneshot_learner = OneShotLearner(omni_glot_reader, config)
  omniglot_oneshot_learner.train()

