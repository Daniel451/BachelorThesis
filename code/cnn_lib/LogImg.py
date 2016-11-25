from collections import OrderedDict
from type_checking import accepts_method
from LogSubplot import LogSubplot
from enum_log_subplot_types import LOG_SUBPLOT_TYPES
from enum_log_data_types import LOG_DATA_TYPES
import numpy as np
from itertools import izip
from matplotlib import use
use("Agg")
from matplotlib import pyplot as plt


class LogImg(object):
    def __init__(self, logpath, log_img_type):
        # maps the LogSubplot objects to its LOG_SUBPLOT_TYPES
        self.log_subplots_map = OrderedDict()

        # saves the enum entry of LOG_IMG_TYPES
        self.log_img_type = log_img_type

        # current iteration / train step to log
        self.train_step = 0

        # saves the path to the logging directory (logpath) + image name (log_img_type.name)
        self.complete_logpath = str()
        self.path_to_logdir = logpath
        self.__set_complete_log_path()


    @accepts_method(LOG_SUBPLOT_TYPES)
    def add_subplot(self, log_subplot_type):
        """
        adds an LogSubplot object to the internal ordered dictionary

        :param log_subplot_type: one entry of enum LOG_SUBPLOT_TYPES
        """
        self.log_subplots_map[log_subplot_type] = LogSubplot(log_subplot_type)


    @accepts_method(LOG_SUBPLOT_TYPES)
    def get_subplot(self, log_subplot_type):
        """
        returns the LogSubplot object corresponding to the specific LOG_SUBPLOT_TYPES enum entry

        :param log_subplot_type: one entry of enum LOG_SUBPLOT_TYPES
        :return: object of LogSubplot
        :rtype: LogSubplot
        """
        return self.log_subplots_map[log_subplot_type]


    @accepts_method(int)
    def set_trainstep(self, new_train_step):
        """
        sets the current training step - must be called for every new image which
        should be generated for a new training step

        :param new_train_step: current training step (int)
        """
        # set the new train step
        self.train_step = new_train_step

        # update the img path
        self.__set_complete_log_path()


    def __set_complete_log_path(self):
        self.complete_logpath = self.path_to_logdir \
                                + self.log_img_type.name \
                                + "_" + "{:0>5}".format(self.train_step)


    def plot(self):
        """
        plots the log image
        """
        subplot_count = len(self.log_subplots_map)

        if subplot_count == 1:
            f, axarr = plt.subplots(1, 1)
            axarr = np.array(axarr)
        elif subplot_count == 2:
            f, axarr = plt.subplots(1, 2)
        elif subplot_count == 3:
            f, axarr = plt.subplots(2, 2)
        else:
            f, axarr = plt.subplots(2, 2)

        # for every subplot zip the corresponding axarr axes and iterate over this
        for ax, subplot in izip(axarr.reshape(np.count_nonzero(axarr)), self.log_subplots_map.itervalues()):

            # set the title for the subplot
            ax.set_title(subplot.get_subplot_type().value.label)

            # plot every data obj of the subplot in the current subplot
            for data_obj in subplot.get_data_values():

                data_dtype = data_obj.get_dtype()
                if data_dtype == LOG_DATA_TYPES.o_img_data:
                    ax.imshow(data_obj.get_data(), interpolation="none", cmap="gray")
                else:
                    ax.plot(data_obj.get_data(),
                            label=data_dtype.value.label,
                            color=data_dtype.value.color.value,
                            marker=data_dtype.value.marker,
                            linestyle=data_dtype.value.linestyle)
                    ax.legend(prop={"size": 6}, loc=2)

        # save the image
        plt.savefig(self.complete_logpath + ".png")

        # clear the plot
        plt.clf()

        # close the plot
        plt.close()




















