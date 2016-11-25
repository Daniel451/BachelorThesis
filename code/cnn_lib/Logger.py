from collections import OrderedDict
from enum_log_img_types import LOG_IMG_TYPES
from type_checking import accepts_method
from LogImg import LogImg
import os


class LoggerClass(object):
    """
    this class manages all logging processes
    """


    def __init__(self, log_path):
        """
        initializes the Logger object

        :param log_path: absolute path to the logging directory
        """
        # create a map for storing LogImg objects
        self.log_img_map = OrderedDict()

        # set the path to the log directory
        self.log_path = log_path

        # check if log directory already exists or create it
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # set current training step
        self.train_step = 0


    @accepts_method(LOG_IMG_TYPES)
    def add_log_img(self, log_img_type):
        """
        Adds a new LogImg object to the logger.

        :param log_img_type: has to be a valid entry of enum LogImgTypes
        """
        self.log_img_map[log_img_type] = LogImg(self.log_path, log_img_type)


    @accepts_method(LOG_IMG_TYPES)
    def get_log_img_obj(self, log_img_type):
        """
        Get an existing log img object reference

        :param log_img_type: one entry of the LOG_IMG_TYPES enum

        :return: reference to LogImg object corresponding to the enum entry
        :rtype: LogImg
        """
        if log_img_type in self.log_img_map:
            return self.log_img_map[log_img_type]
        else:
            msg = "error: log_img_type '{}' does not exist in the Logger object.\n".format(log_img_type)
            msg += "There are currently {} objects saved in the Logger object".format(len(self.log_img_map))
            exit(msg)


    @accepts_method(int)
    def set_train_step(self, new_train_step):
        """
        sets the current training step - must be called for every new image which
        should be generated for a new training step

        :param new_train_step: current training step (int)
        """
        self.train_step = new_train_step

        # update all childrens -> every LogImg object saved in the map
        for logimg in self.log_img_map.itervalues():
            logimg.set_trainstep(self.train_step)


    def plot_data(self):
        """
        starts plotting of all previously saved data
        """
        # plot every log image
        for log_img in self.log_img_map.itervalues():
            log_img.plot()
