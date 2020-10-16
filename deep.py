
# Lab 11 MNIST and Convolutional Neural Network
import math
import os
import time
from math import ceil

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from PIL import Image

import datetime
import io
import random

np.set_printoptions(threshold=np.nan)

# import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility


# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

class Network:
    def __init__(self, layers=None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        # hyper parameters
        learning_rate = 0.001
        training_epochs = 15
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, 128, 128])
        X_img = tf.reshape(self.X, [-1, 128, 128, 1])   # img 128x128x1 (black/white)
        self.Y = tf.placeholder(tf.float32, [None, 20])
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        # L1 ImgIn shape=(?, 128, 128, 1)
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        #    Conv     -> (?, 128, 128, 32)
        #    Pool     -> (?, 64, 64, 32)
        L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        '''
        Tensor("Conv2D:0", shape=(?, 128, 128, 32), dtype=float32)
        Tensor("Relu:0", shape=(?, 128, 128, 32), dtype=float32)
        Tensor("MaxPool:0", shape=(?, 64, 64, 32), dtype=float32)
        '''
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        #    Conv     -> (?, 64, 64, 64)
        #    Pool     -> (?, 32, 32, 64)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        # L3 ImgIn shape=(?, 32, 32, 64)
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 96], stddev=0.01))
        #    Conv      ->(?, 32, 32, 96)
        #    Pool      ->(?, 16, 16, 96)
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L3_flat = tf.reshape(L3, [-1, 16 * 16 * 96])
        '''
        Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
        Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
        '''

        # Final FC 7x7x64 inputs -> 20 outputs
        W4 = tf.Variable(tf.random_normal([16 * 16 * 96, 20],stddev=0.01))
        b = tf.Variable(tf.random_normal([20]))
        self.logits = tf.matmul(L3_flat, W4) + b
        self.description = ""
        self.cost = tf.reduce_mean(tf.square(self.logits -self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.logits)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.Y), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            #tf.summary.scalar('accuracy', self.accuracy)

        #self.summaries = tf.summary.merge_all()
    
class Dataset:
    def __init__(self, batch_size, folder='voc'):
        self.batch_size = batch_size

        train_inputs, validation_inputs = self.train_valid_split(
            os.listdir(os.path.join(folder, 'inputs')))
        train_targets, validation_targets = self.train_valid_split(
            os.listdir(os.path.join(folder, 'targets')))
        self.train_inputs = self.file_paths_to_input(folder, train_inputs)
        self.validation_inputs = self.file_paths_to_input(folder, validation_inputs)
        self.train_targets = self.file_paths_to_target(folder, train_targets)
        self.validation_targets = self.file_paths_to_target(folder, validation_targets)

        self.pointer = 0

    def file_paths_to_input(self, folder, files_list, verbose=False):
        inputs = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs', file)
            test_image = Image.open(input_image)
            test_image = test_image.convert("L") # load grayscale
            test_image = test_image.resize((128,128))
            test_image = np.array(test_image)
            # test_image = np.multiply(test_image, 1.0 / 255)
            inputs.append(test_image)

        return inputs
    
    def file_paths_to_target(self, folder, files_list, verbose=False):
        targets = []

        for file in files_list:
            targets_array = os.path.join(folder, 'targets', file)
            # test_image = np.multiply(test_image, 1.0 / 255)
            targets_array = np.genfromtxt(targets_array)
            targets_array=np.array(targets_array)
            targets.append(targets_array)
            
        return targets    

    def train_valid_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.75, .25)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])):]
        )
    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.RandomState(seed=42).permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))
        self.pointer += self.batch_size


        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def validation_set(self):
        return np.array(self.validation_inputs, dtype=np.uint8), np.array(self.validation_targets, dtype=np.uint8)

class train():
    # hyper parameters
    learning_rate = 0.001
    batch_size = 100
    network = Network()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    
    #os.makedirs(os.path.join('save', network.description, timestamp))

    dataset = Dataset(folder='voc',
                      batch_size=batch_size)


    # initialize
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
        #                                        graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
        test_accuracies = []
        training_epochs = 15
        global_start=time.time()
        # train my model
        print('Learning started. It takes sometime.')
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = dataset.num_batches_in_epoch()
            dataset.reset_batch_pointer()
            for i in range(total_batch):
                batch_inputs, batch_targets = dataset.next_batch()
                batch_num = epoch * dataset.num_batches_in_epoch() + i + 1
                batch_inputs = np.reshape(batch_inputs,
                        (dataset.batch_size, 128, 128))
                batch_targets = np.reshape(batch_targets,
                        (dataset.batch_size, 20))
                feed_dict = {network.X: batch_inputs, network.Y: batch_targets}
                c, _ = sess.run([network.cost, network.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
                end = time.time()
                if batch_num % 100 == 0 or batch_num == training_epochs * dataset.num_batches_in_epoch():
                    validation_inputs, validation_targets = dataset.validation_set
                    # test_inputs, test_targets = test_inputs[:100], test_targets[:100]

                    validation_inputs = np.reshape(validation_inputs, (-1, 128, 128))
                    validation_targets = np.reshape(validation_targets, (-1, 20))
                   

                    validation_accuracy = sess.run(network.accuracy,
                                                        feed_dict={network.X: validation_inputs,
                                                                network.Y: validation_targets,
                                                                network.is_training: False})

                    #summary_writer.add_summary(summary, batch_num)

                    print('Step {}, test accuracy: {}'.format(batch_num, validation_accuracy))
                    test_accuracies.append((validation_accuracy, batch_num))
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))

                    if validation_accuracy >= max_acc[0]:
                        checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=batch_num)


            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')


if __name__ == '__main__':
    train()
