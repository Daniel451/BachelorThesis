import time
import tensorflow as tf
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from model_world import World
from custom_rnn_cell import CustomRNNCell


class RNN(object):

    hidden_size = None
    cell = None
    inputs = None
    inputs_lst = None
    targets = None
    h_to_o_b = None
    h_to_o_W = None
    logits = None
    loss = None
    inference = None
    experiment = None         # string; name of the experiment
    acti_func = None          # activation function of the network
    log_every_n_steps = None  # logging interval
    logpath = None            # path, were the logfiles should be stored

    def __init__(self, experiment="sample_ex",
                        hidden_size=100,
                        batch_size=1,
                        activation_function=tf.nn.tanh,
                        log_every_n_steps=1000,
                        learning_rate=1e-4):
        """
        constructor for the recurrent neural network
        - recurrent timesteps are always exactly 1 at the moment

        :param hidden_size: defines the amount of hidden neurons
        :param batch_size: batch_size of the features, for the RNN always 1
        :param activation_function: the activation (transfer) function to use, default is hyperbolic tangent (tf.nn.tanh)
        """
        self.experiment = experiment
        self.logpath = str(os.environ["BTLOG"]) + self.experiment + "/"

        # create logging directory, if it does not exist
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)

        self.log_every_n_steps = int(log_every_n_steps)
        self.learning_rate = learning_rate

        self.hidden_size = hidden_size
        self.acti_func = activation_function
        self.cell = CustomRNNCell(self.hidden_size, self.acti_func) 
        self.inputs = tf.placeholder(tf.float32, shape=[batch_size, 800], name="inputs")

        # The RNN Interface requires that images are split into a list of [batch_size x hidden_size]
        #self.inputs_lst = [tf.squeeze(indat, [1]) for indat in tf.split(1, 800, self.inputs)]
        self.inputs_lst = [self.inputs]
        #print(self.inputs_lst[0])
        self.targets = tf.placeholder(tf.float32, shape=[batch_size, 800], name="targets")


    def build_model(self):
        with tf.variable_scope("rnn"):
            output, state = tf.nn.rnn(self.cell, self.inputs_lst, dtype=tf.float32)
            #self.h_to_o_W = tf.get_variable("h_to_o_W", [self.hidden_size, 800],
                                            #initializer=tf.truncated_normal_initializer(0.01))
            self.h_to_o_W = self.xavier_init((self.hidden_size, 800))
            self.h_to_o_b = tf.get_variable("h_to_o_b", [800],
                                            initializer=tf.constant_initializer(0.0))

            self.logits = tf.matmul(output[-1], self.h_to_o_W) + self.h_to_o_b

            self.logits_softmax = tf.nn.softmax(self.logits)

            loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets)

            self.loss = tf.reduce_mean(loss)

            self.inference = tf.argmax(tf.nn.softmax(self.logits), dimension=1)

            self.top11_out = tf.nn.top_k(self.logits, 11)[1]
            self.top11_train = tf.nn.top_k(self.targets, 11)[1]

            # initialize the graph/session
            self.sess = tf.Session()
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.sess.run(tf.initialize_all_variables())

    
    def xavier_init(self, shape):
        # for fully connected layers
        n_in, n_out = tf.cast(shape[0], dtype=tf.float32), tf.cast(shape[1], dtype=tf.float32)

        # calculate numerator and denominator
        xavier_numerator = tf.sqrt(6.0)
        xavier_denominator = tf.sqrt(tf.add(n_in, n_out))

        # calculate the xavier variance
        xavier_stddev = tf.div(xavier_numerator, xavier_denominator)

        # initialize with uniform distribution
        initial = tf.random_uniform(shape, minval=-xavier_stddev, maxval=xavier_stddev, dtype=tf.float32)

        return tf.Variable(initial)


    def load_network(self, loadpath=""):
        """
        load a previously trained network

        :param loadpath: (optional!) path, from where the network data should be loaded
        """
        # manipulate the path from where the network should be loaded
        if loadpath == "":
            loadpath = self.logpath + "aab_model.ckpt"

        saver = tf.train.Saver()
        saver.restore(self.sess, loadpath)


    def train(self, num_iterations=10000, visualize_error=False):
        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []

        # convert
        num_iterations = int(num_iterations)

        # world model
        W = World()

        # object for saving the model later on
        saver = tf.train.Saver()

        # clear the log directory
        for f in os.listdir(self.logpath):
            fpath = os.path.join(self.logpath, f)
            if os.path.isfile(fpath) and "aaa_out.txt" not in fpath:
                os.unlink(fpath)

        ###################
        # actual training #
        ###################
        for i in range(num_iterations):
            W.update_world()
            trX, trY = W.get_last_sensor_data_noise(), W.get_current_sensor_data() 
            tr_feed_dict = {self.inputs: trX, self.targets: trY}

            self.sess.run(self.train_op, feed_dict=tr_feed_dict)

            train_loss, train_pred = self.sess.run([self.loss, self.inference],
                                              feed_dict=tr_feed_dict)

            # only append every n-th (log_every_n_steps) loss
            if i % self.log_every_n_steps == 0:
                train_losses.append(train_loss)

            # visualization and printing stuff (for learning)
            if visualize_error and i % self.log_every_n_steps == 0:
                plt.clf()
                plt.plot(train_losses)
                plt.pause(1e-10)
                print("\niter {}:".format(i))
                print("train error: {: >2.4f}".format(train_loss))

            # log learning process
            if i % self.log_every_n_steps == 0:
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.plot(train_losses)
                f.savefig(self.logpath + "error" + ".png")
                f.clf()
                plt.close()

        # save model, after training is complete
        saver.save(self.sess, self.logpath + "aab_model.ckpt")


    def visual_evaluation(self):
        # visual evaluation
        plot_rows = 2
        plot_cols = 2

        f = plt.figure()

        data = plt.subplot2grid((plot_rows, plot_cols), (0, 0), colspan=2, aspect="auto")
        #top11 = plt.subplot2grid((plot_rows, plot_cols), (1, 0), colspan=2, aspect="auto")

        W = World()

        for i in range(1000):
            # get new world data
            W.update_world()
            trX, trY = W.get_last_sensor_data_noise(), W.get_current_sensor_data() 
            
            # send data to trained network and get data back
            nno, top11_out, top11_train = self.sess.run([self.logits_softmax, self.top11_out, self.top11_train],
                    feed_dict={
                        self.inputs: trX,
                        self.targets: trY
                })
           
            # unpack data
            nno = nno[0]
            top11_out = top11_out[0]
            top11_train = top11_train[0]
            trX = trX[0]
            trY = trY[0]
        
            ############
            # plotting #
            ############
            data.cla()
            data.plot(nno, color="blue", label="network output")
            data.plot(trX, color="orange", label="X")
            data.plot(trY, color="red", label="Y")
            data.legend() 
            
            #top11.cla()
            #top11.plot(top11_out, linestyle="none", marker="x", color="magenta", label="top11 out")
            #top11.plot(top11_out, linestyle="none", marker="o", color="black", label="top11 train")
            #top11.legend()

            f.show()
            plt.pause(1.0)



