from LogData import LogData
from type_checking import accepts_method
from enum_log_data_types import LOG_DATA_TYPES
import numpy as np


class DataVector(LogData):
    def __init__(self, data_type):
        super(DataVector, self).__init__()

        # data
        self.data = None
        self.dtype = data_type


    def add_data(self, values):
        """
        - for single data vectors -
        takes one data vector and saves it (flushes old values)

        :param values: data (peferably a list or numpy vector, not an array)
        """
        self.data = np.array(values)
