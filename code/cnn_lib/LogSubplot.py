from LogData import LogData
from enum_log_data_types import LOG_DATA_TYPES
from type_checking import accepts_method
from collections import OrderedDict


class LogSubplot(object):

    def __init__(self, log_subplot_type):
        self.log_data_objects = OrderedDict()
        self.subplot_type = log_subplot_type


    @accepts_method(LogData, LOG_DATA_TYPES)
    def add_data_obj(self, log_data_obj, log_data_type):
        """
        adds a new data obj for current LogSubplot object

        :param log_data_obj: one LogData object
        :param log_data_type: one entry of enum LOG_DATA_TYPES
        """
        self.log_data_objects[log_data_type] = log_data_obj


    def get_data_values(self):
        """
        :return: returns an OrderedDict of LogData objects
        :rtype: OrderedDict
        """
        return self.log_data_objects.itervalues()


    def get_subplot_type(self):
        return self.subplot_type


    @accepts_method(LOG_DATA_TYPES)
    def get_data_obj(self, log_data_type):
        """
        returns the specified LogData object

        :param log_data_type: one entry of enum LOG_DATA_TYPES
        :return: a LogData object
        :rtype: LogData
        """
        return self.log_data_objects[log_data_type]

