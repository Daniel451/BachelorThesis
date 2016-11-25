from LogData import LogData
from type_checking import accepts_method
from enum_log_data_types import LOG_DATA_TYPES
import numpy as np


class DataSequence(LogData):
    def __init__(self, data_type):
        super(DataSequence, self).__init__()

        # data
        self.data = list()
        self.dtype = data_type


    def add_data(self, value):
        """
        - for creating data sequences over time -
        takes one data value and appends it to the list of data (never flushed)

        :param value: one data value, e.g. a float, int, ...
        """
        self.data.append(value)


    def get_data(self):
        """
        :return: returns a tuple -> sequence length + the data
        :rtype: tuple(range(len(sequence)), numpy.ndarray)
        """
        return np.array(self.data)


