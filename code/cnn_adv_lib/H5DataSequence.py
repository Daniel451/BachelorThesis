import numpy as np


class H5DataSequence(object):
    def __init__(self):

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


    def get_number_of_entries(self):
        """
        :return: returns the length (int) of the sequence's entries
        """
        return len(self.data)


