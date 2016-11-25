import abc
import numpy as np
from type_checking import accepts_method


class LogData(object):
    __metaclass__ = abc.ABCMeta


    def __init__(self):
        self.data = None
        self.dtype = None
        self.plot_kwargs = None


    @abc.abstractmethod
    def add_data(self, value):
        """
        takes input value(s) and adds it to a buffer
        :param value:
        """


    @accepts_method(dict)
    def set_kwargs_for_plotting(self, kwargs_dict):
        """
        sets the kwargs for matplotlib plt.plot()

        :param kwargs_dict: kwargs formatted as one dictionary
        :type kwargs_dict: dict
        """
        self.plot_kwargs = kwargs_dict


    def get_kwargs_for_plotting(self):
        """
        returns the **kwargs for matplotlib plt.plot()
        """
        return self.plot_kwargs


    def get_dtype(self):
        """
        :return: returns the type of data -> enum entry
        :rtype: LOG_DATA_TYPES
        """
        return self.dtype


    def get_data(self):
        """
        :return: returns the data
        :rtype: numpy.ndarray
        """
        return np.array(self.data)
