# dataset stuff
from cnn_lib.BufferDataLoader import DataObject

# enumeration stuff
from cnn_lib.enum_colors import COLORS
from cnn_lib.enum_log_img_types import LOG_IMG_TYPES
from cnn_lib.enum_log_subplot_types import LOG_SUBPLOT_TYPES
from cnn_lib.enum_log_data_types import LOG_DATA_TYPES

# log stuff
from cnn_lib.DataSequence import DataSequence
from cnn_lib.DataVector import DataVector
from cnn_lib.Logger import LoggerClass

# debug stuff
from extra.array_type_printer import ArrayType as AT

# python libs
from itertools import cycle, izip
from collections import OrderedDict

# system variables and shell stuff
import sys, os, shutil

# tensorflow library
import tensorflow as tf

# numpy library
import numpy as np

# save numpy arrays / data
import h5py

# time for execution time measurements
import time






#####################
# actual experiment #
#####################
class abstr_model_s11(object):
    """
    abstract model
    """


    def __init__(self): 
        # dimensions of image data
        self.img_x_size = 800
        self.img_y_size = 600

        # just for training
        self.current_training_iteration = 0

        # log_path has to be overwritten
        self.log_path = str(os.environ["BTLOG"]) + self.experiment + "/"

        raise NotImplementedError("this is an abstact class; implementation needed")


    def run_network(self, data_input):
        """
        computes one feed-forward run with given input data

        :param data_input: input data; has to be a tensor of shape (batch_size, 600, 800, 3)
        """
        nn_out_x, nn_out_y,\
        nn_out_x_softmax, nn_out_y_softmax,\
        top11_x, top11_y = self.sess.run(
                [self.nn_out_x_linear, self.nn_out_y_linear,
                 self.nn_out_x_softmax, self.nn_out_y_softmax,
                 self.top11_x_out, self.top11_y_out],
                feed_dict={
                    self.input_x: data_input,
                    self.keep_prob: 1.0
                }
        )

        return nn_out_x, nn_out_y, nn_out_x_softmax, nn_out_y_softmax, top11_x, top11_y


    def load_network(self, path=""):
        """
        loads a previously trained network via tensorflow checkpoints

        :param path: path to the tensorflow checkpoint file, defaults to the default log directory
        """
        # if no specific path is given...
        if path == "":
            path = self.log_path + "aab_model.ckpt" # ...load data from default directory

        saver = tf.train.Saver()

        self.print_with_time("loading network data from checkpoint '{}'...".format(path))
        saver.restore(self.sess, path)
        self.print_with_time("successfully loaded graph from stored model!")


    def build_evaluation(self):
        ##################
        # train and eval #
        ##################

        # top11 [0] -> values, [1] -> args
        self.top11_x_out = tf.nn.top_k(self.nn_out_x_linear, 11)[1]
        self.top11_y_out = tf.nn.top_k(self.nn_out_y_linear, 11)[1]
        self.top11_x_train = tf.nn.top_k(self.input_y_train_x, 11)[1]
        self.top11_y_train = tf.nn.top_k(self.input_y_train_y, 11)[1]

        # cross entropy
        self.cross_entropy_x = tf.nn.softmax_cross_entropy_with_logits(self.nn_out_x_linear, self.input_y_train_x)
        self.cross_entropy_y = tf.nn.softmax_cross_entropy_with_logits(self.nn_out_y_linear, self.input_y_train_y)
        self.cross_entropy = (self.cross_entropy_x + self.cross_entropy_y) / 2.0

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        self.abs_distances_x = tf.abs(tf.argmax(self.nn_out_x_linear, 1) - tf.argmax(self.input_y_train_x, 1))
        self.accuracy_x = tf.reduce_mean(self.abs_distances_x) 

        self.abs_distances_y = tf.abs(tf.argmax(self.nn_out_y_linear, 1) - tf.argmax(self.input_y_train_y, 1))
        self.accuracy_y = tf.reduce_mean(self.abs_distances_y) 


    def __build_model(self):
        """
        sets up the network -> graph/model 
        """
        raise NotImplementedError("this method builds the actual model; has to be overwritten")
        

    def __check_dataset_objects(self, dataset_objects):
        """
        :param dataset_objects: a list of dataset objects which provide the training/test data and labels
        """
        if type(dataset_objects) is not list:
            raise TypeError("argument was '{}' but expected type is 'list'".format(type(dataset_objects)))

        if not len(dataset_objects) > 0:
            raise Exception("list has to contain at least one dataset object")

        for e in dataset_objects:
            if type(e) is not DataObject:
                raise TypeError("list must only contain dataset objects, "\
                                + "but one item had the type '{}'".format(type(e)))


    def train(self,
              train_data_object_cycler,
              test_dataset,
              log_every_n_steps=1000,
              amount_of_train_steps=1e5,
              dropout_rate=1.0):
        """
        :param train_dataset_objects: a list of dataset objects which provide the training/test data and labels
        :param test_dataset: one dataset object containing the test data to evaluate while training
        :param log_every_n_steps: defines the steps at which a logging process should be run
        :param amount_of_train_steps: total amount of training steps to execute
        :param dropout_rate: float in range [0.0, 1.0]
                                -> 0.0: 100% dropout chance for every neuron
                                -> 0.5:  50% dropout chance
                                -> 1.0:   0% dropout chance (-> no dropout at all)
        """
        self.print_with_time("start training...")
        #######################################
        # check and initialize training stuff #
        #######################################
        train_dataset_objects = train_data_object_cycler.get_all_data_objects()

        # check for valid dataset objects
        self.__check_dataset_objects(train_dataset_objects)
        self.__check_dataset_objects([test_dataset])

        # prepare for a new training session
        self.__initialize_logging_for_training(log_every_n_steps, amount_of_train_steps)

        self.log_every_n_steps = int(log_every_n_steps)
        self.amount_of_train_steps = int(amount_of_train_steps)

        ###################
        # actual training #
        ###################
        total_start_time = int(time.time())

        for i in xrange(1, amount_of_train_steps + 1):
            # set current training iteration
            self.current_training_iteration = i

            # cycle through the dataset objects
            data = train_data_object_cycler.get_next_dataset_for_batch_loading() 

            # get the next batch for training
            tr_imgdata, tr_Y_x, tr_Y_y = data.get_next_batch()

            # logging and accuracy
            if i % self.log_every_n_steps == 0:
                # evaluation
                self.__evaluate_current_training_status(test_dataset)

                # plot execution time
                exec_time_delta = int(time.time() - total_start_time)
                exec_minutes, exec_seconds = divmod(exec_time_delta, 60)
                exec_hours, exec_minutes = divmod(exec_minutes, 60)
                self.print_with_time("training step: {:>6} | train exec_time: {:0>2}:{:0>2}:{:0>2}"\
                                     .format(i, exec_hours, exec_minutes, exec_seconds))


            # save the current model every 10%
            if int(round(i % int(self.amount_of_train_steps / 10.0))) == 0:
                # save current model
                self.print_with_time("training step: {:>6} | saving current network progress...".format(i))
                saver = tf.train.Saver()
                saver.save(self.sess, self.log_path + "aab_model.ckpt")
                self.print_with_time("finished saving progress")


            # actual training step
            self.sess.run(self.train_step,
                          feed_dict={self.input_x: tr_imgdata,
                                     self.input_y_train_x: tr_Y_x,
                                     self.input_y_train_y: tr_Y_y,
                                     self.keep_prob: dropout_rate})


        #######################
        # total time consumed #
        #######################
        total_time_delta = int(time.time() - total_start_time)
        total_minutes, total_seconds = divmod(total_time_delta, 60)
        total_hours, total_minutes = divmod(total_minutes, 60)

        print("------------------------------------------")
        self.print_with_time("training finished!")
        self.print_with_time("total training time: {:0>2}:{:0>2}:{:0>2}"\
                             .format(total_hours, total_minutes, total_seconds))
        print("({} training steps)".format(self.amount_of_train_steps))

        # reset current training iteration
        self.current_training_iteration = 0

        #####################
        # finished training #
        #####################
        # save the model
        print("")
        self.print_with_time("start saving final network model...")

        # completely save the model
        saver = tf.train.Saver()
        saver.save(self.sess, self.log_path + "aab_model.ckpt")

        # save the convolutional layers
        conv_saver = tf.train.Saver({
                                    "W_conv1": self.W_conv1,
                                    "b_conv1": self.b_conv1,
                                    "W_conv2": self.W_conv2,
                                    "b_conv2": self.b_conv2,
                                    "W_conv3": self.W_conv3,
                                    "b_conv3": self.b_conv3,
                                    })
        conv_saver.save(self.sess, self.log_path + "aab_model_convolutions.ckpt")

        self.print_with_time("finished saving!")


    def evaluate_test_data_set(self):
        """
        automatically evaluates the network with the test data set
        """
        raise NotImplementedError()


    def __evaluate_current_training_status(self, test_dataset):
        #####################################
        # accuracy testing on test data set #
        #####################################
        top11_x_container = list()
        top11_y_container = list()

        accuracy_container_x = list()
        accuracy_container_y = list()

        ce_container_x = list()
        ce_container_y = list()

        # evaluate the whole test dataset
        for i in xrange(test_dataset.get_dataset_size()):
            # extract next img
            te_imgdata, te_Y_x, te_Y_y, _fname = test_dataset.get_specific_image(i)

            # run the network
            accu_x, accu_y,\
            r_top11_x_out, r_top11_y_out,\
            r_top11_x_train, r_top11_y_train,\
            te_nn_out_x, te_nn_out_y,\
            ce_x, ce_y = self.sess.run(
                    [self.accuracy_x, self.accuracy_y,\
                     self.top11_x_out, self.top11_y_out,\
                     self.top11_x_train, self.top11_y_train,\
                     self.nn_out_x_linear, self.nn_out_y_linear,\
                     self.cross_entropy_x, self.cross_entropy_y],
                    feed_dict={
                        self.input_x: te_imgdata,
                        self.input_y_train_x: [te_Y_x],
                        self.input_y_train_y: [te_Y_y],
                        self.keep_prob: 1.0
                    }
            )

            # cross entropy x/y
            ce_container_x.append(ce_x)
            ce_container_y.append(ce_y)

            # accuracy x/y
            accuracy_container_x.append(accu_x)
            accuracy_container_y.append(accu_y)

            # top11
            for x11, y11,\
                _x11, _y11 in\
                    izip(r_top11_x_out, r_top11_y_out,\
                         r_top11_x_train, r_top11_y_train):
                top11_x_container.append(
                        np.mean(np.in1d(x11, _x11).astype(np.float32))
                )
                top11_y_container.append(
                        np.mean(np.in1d(y11, _y11).astype(np.float32))
                )

        ######################
        ### total accuracy ###
        ######################
        total_accuracy_x_mean = np.mean(accuracy_container_x)
        total_accuracy_y_mean = np.mean(accuracy_container_y)
        
        total_accuracy_x_median = np.median(accuracy_container_x)
        total_accuracy_y_median = np.median(accuracy_container_y)

        ####################
        ### total top-11 ###
        ####################
        total_top11_x_mean = np.mean(top11_x_container)
        total_top11_y_mean = np.mean(top11_y_container)

        total_top11_x_median = np.median(top11_x_container)
        total_top11_y_median = np.median(top11_y_container)

        #####################
        ### cross entropy ###
        #####################
        total_ce_x_mean = np.mean(ce_container_x)
        total_ce_y_mean = np.mean(ce_container_y)

        total_ce_x_median = np.median(ce_container_x)
        total_ce_y_median = np.median(ce_container_y)


        #######################
        ### write hdf5 data ###
        #######################
        # write accuracy and top11 test accuracy to disk
        h5f_filepath = "{}training_log_data.h5".format(self.log_path)

        h5f = h5py.File(h5f_filepath, "a")

        h5f["testdata"]["top11_x_mean"][self.h5f_index] = total_top11_x_mean
        h5f["testdata"]["top11_y_mean"][self.h5f_index] = total_top11_y_mean
        h5f["testdata"]["top11_x_median"][self.h5f_index] = total_top11_x_median
        h5f["testdata"]["top11_y_median"][self.h5f_index] = total_top11_y_median

        h5f["testdata"]["accu_x_mean"][self.h5f_index] = total_accuracy_x_mean 
        h5f["testdata"]["accu_y_mean"][self.h5f_index] = total_accuracy_y_mean 
        h5f["testdata"]["accu_x_median"][self.h5f_index] = total_accuracy_x_median 
        h5f["testdata"]["accu_y_median"][self.h5f_index] = total_accuracy_y_median

        h5f["testdata"]["ce_x_mean"][self.h5f_index] = total_ce_x_mean 
        h5f["testdata"]["ce_y_mean"][self.h5f_index] = total_ce_y_mean 
        h5f["testdata"]["ce_x_median"][self.h5f_index] = total_ce_x_median 
        h5f["testdata"]["ce_y_median"][self.h5f_index] = total_ce_y_median 

        h5f.close()

        # increase h5f index, otherwise the next evaluation data
        # would overwrite current data
        self.h5f_index += 1


    def __initialize_logging_for_training(self, log_every_n_steps, amount_of_train_steps):
        """
        prepares everything for a new training session
        """
        ###################
        # clean directory #
        ###################
        num_log_data_entries = int(amount_of_train_steps / log_every_n_steps)

        # internal index, used for pointing at the index of
        # the hd5f logging file
        self.h5f_index = 0

        # delete all files in the log directory
        # to have a fresh directory for the new training
        for f in os.listdir(self.log_path):
            fpath = os.path.join(self.log_path, f)
            if os.path.isfile(fpath) and "aaa_out.txt" not in fpath:
                os.unlink(fpath)

        # creates a new hdf5 data file -- or overwrites an existing one
        h5f_filepath = "{}training_log_data.h5".format(self.log_path)
        h5f = h5py.File(h5f_filepath, "w")

        h5f.attrs["file_name"] = h5f_filepath.split("/")[-1] 
        h5f.attrs["file_time"] = time.strftime("[%d.%m.%y | %H:%M:%S]", time.gmtime())
        h5f.attrs["file_timestamp"] = int(time.time())
        h5f.attrs["creator"] = self.experiment
        h5f.attrs["HDF5_version"] = h5py.version.hdf5_version
        h5f.attrs["h5py_version"] = h5py.version.version
        h5f.attrs["log_every_n_steps"] = log_every_n_steps
        h5f.attrs["amount_of_train_steps"] = amount_of_train_steps 

        h5f.create_group("testdata")

        h5f["testdata"].create_dataset("top11_x_mean", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("top11_y_mean", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("top11_x_median", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("top11_y_median", shape=(num_log_data_entries,), dtype=np.float32)

        h5f["testdata"].create_dataset("accu_x_mean", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("accu_y_mean", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("accu_x_median", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("accu_y_median", shape=(num_log_data_entries,), dtype=np.float32)

        h5f["testdata"].create_dataset("ce_x_mean", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("ce_y_mean", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("ce_x_median", shape=(num_log_data_entries,), dtype=np.float32)
        h5f["testdata"].create_dataset("ce_y_median", shape=(num_log_data_entries,), dtype=np.float32)

        h5f.close()



    def print_with_time(self, cstr):
        timestr = time.strftime("[%d.%m.%y | %H:%M:%S]", time.gmtime())
        print("{} {}".format(timestr, cstr))
        sys.stdout.flush()


