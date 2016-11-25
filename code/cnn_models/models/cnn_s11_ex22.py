# dataset stuff
from cnn_lib.BufferDataLoader import DataObject

# debug stuff
from extra.array_type_printer import ArrayType as AT

# network stuff
from cnn_models.models.abstr_model import abstr_model_s11 as abstr_net

# python libs
from itertools import cycle, izip
from collections import OrderedDict

# system variables and shell stuff
import os, shutil

# tensorflow library
import tensorflow as tf

# numpy library
import numpy as np

# time for execution time measurements
import time






#####################
# actual experiment #
#####################
class NetworkModel(abstr_net):
    """
    new series 10 for dynamic work
    -> model itself mostly cloned from s07e17

    full-size (800x600 imgs)
    """


    def __init__(self, experiment_name): 
        self.print_with_time("initializing network...")
        self.experiment = experiment_name 

        # initiialize the abstract class
        try:
            super(NetworkModel, self).__init__()
        # NotImplementedError is thrown by abstract class when initializing it
        # there we catch and proceed with our NetworkModel
        except NotImplementedError:
            self.print_with_time("successfully initialized abstract model!")
        else:
            raise Exception("something went wrong initializing abstract model...")

        #######################
        # configuration stuff #
        #######################
        self.learning_rate = 1e-6

        # dimensions of image data
        self.img_x_size = 200
        self.img_y_size = 150

        #####################
        # AI initialization #
        #####################
        self.__build_model()


    def __build_model(self):
        """
        sets up the network -> graph/model 
        """
        self.print_with_time("start building up model '{}'...".format(self.experiment))

        ################################################################################
        # weight, bias, conv2d and max_pool methods for initialization and calculation #
        ################################################################################
        def weight_variable(shape, name):
            if len(shape) == 2:
                # for fully connected layers
                n_in, n_out = tf.cast(shape[0], dtype=tf.float32), tf.cast(shape[1], dtype=tf.float32)
            else:
                # for convolutional layers
                n_in = tf.cast(shape[0], tf.float32) \
                       * tf.cast(shape[1], tf.float32) \
                       * tf.cast(shape[2], tf.float32)

                n_out = tf.cast(shape[3], tf.float32)

            # calculate numerator and denominator
            xavier_numerator = tf.sqrt(6.0)
            xavier_denominator = tf.sqrt(tf.add(n_in, n_out))

            # calculate the xavier variance
            xavier_stddev = tf.div(xavier_numerator, xavier_denominator)

            # initialize with uniform distribution
            initial = tf.random_uniform(shape, minval=-xavier_stddev, maxval=xavier_stddev, dtype=tf.float32)

            return tf.Variable(initial)


        def bias_variable(shape, name, start_value=0.01):
            initial = tf.constant(start_value, shape=shape, dtype=tf.float32)
            return tf.Variable(initial)


        def conv2d(x2d, W):
            return tf.nn.conv2d(x2d, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(mx):
            return tf.nn.max_pool(mx, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        def max_pool_8x8(mx):
            return tf.nn.max_pool(mx, ksize=[1, 8, 8, 1],
                                  strides=[1, 8, 8, 1], padding='SAME')

        def avg_pool_2x2(mx):
            return tf.nn.avg_pool(mx, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        def activation(x):
            return tf.nn.tanh(x) 


        ##############################
        # placeholder for data input #
        ##############################
        self.input_x = tf.placeholder("float", shape=[None, self.img_y_size, self.img_x_size, 3])
        self.input_y_train_x = tf.placeholder("float", shape=[None, self.img_x_size])
        self.input_y_train_y = tf.placeholder("float", shape=[None, self.img_y_size])

        # reshape data
        x_image = tf.reshape(self.input_x, [-1, self.img_y_size, self.img_x_size, 3])

        ########################
        # network architecture #
        ########################
        self.keep_prob = tf.placeholder("float")

        # conv/maxpool layer
        W_conv1 = weight_variable([15, 15, 3, 16], "W_conv1")
        b_conv1 = bias_variable([16], "b_conv1")

        h_conv1 = activation(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_pool1 = tf.nn.dropout(h_pool1, self.keep_prob)

        # conv/maxpool layer
        W_conv2 = weight_variable([7, 7, 16, 32], "W_conv2")
        b_conv2 = bias_variable([32], "b_conv2")

        h_conv2 = activation(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)
        h_pool2 = tf.nn.dropout(h_pool2, self.keep_prob)

        # conv/maxpool layer
        W_conv3 = weight_variable([3, 3, 32, 32], "W_conv3")
        b_conv3 = bias_variable([32], "b_conv3")

        h_conv3 = activation(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = h_conv3
        h_pool3 = tf.nn.dropout(h_pool3, self.keep_prob)

        # flat data for fully connected layers
        flat_size = 38 * 50 * 32  # flattened size of the last conv layer output
        h_pool_flat = tf.reshape(h_pool3, [-1, flat_size])

        # fully connected layer 1 - x
        W_fc1_x = weight_variable([flat_size, 400], "W_fc1_x")
        b_fc1_x = bias_variable([400], "b_fc1_x", start_value=0.0)

        h_fc1_x = activation(tf.matmul(h_pool_flat, W_fc1_x) + b_fc1_x)

        h_fc1_drop_x = tf.nn.dropout(h_fc1_x, self.keep_prob)

        # OUTPUT: fully connected - x
        W_fc2_x = weight_variable([400, self.img_x_size], "W_fc2_x")
        b_fc2_x = bias_variable([self.img_x_size], "b_fc2_x", start_value=-0.1)

        self.nn_out_x_linear = tf.matmul(h_fc1_drop_x, W_fc2_x) + b_fc2_x
        self.nn_out_x_softmax = tf.nn.softmax(self.nn_out_x_linear)

        # fully connected layer 1 - y
        W_fc1_y = weight_variable([flat_size, 300], "W_fc1_y")
        b_fc1_y = bias_variable([300], "b_fc1_y", start_value=0.0)

        h_fc1_y = activation(tf.matmul(h_pool_flat, W_fc1_y) + b_fc1_y)

        h_fc1_drop_y = tf.nn.dropout(h_fc1_y, self.keep_prob)

        # OUTPUT: fully connected - y
        W_fc2_y = weight_variable([300, self.img_y_size], "W_fc2_y")
        b_fc2_y = bias_variable([self.img_y_size], "b_fc2_y", start_value=-0.1)

        self.nn_out_y_linear = tf.matmul(h_fc1_drop_y, W_fc2_y) + b_fc2_y
        self.nn_out_y_softmax = tf.nn.softmax(self.nn_out_y_linear)

        # setting up evaluation stuff (accuracy, top-11 error)
        self.build_evaluation()

        #################
        # session stuff #
        #################
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

        self.print_with_time("finished building up model!")

