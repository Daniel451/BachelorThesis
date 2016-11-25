from matplotlib import image as mpimg
import numpy as np
import os


class AbstractDataObject(object):
    def __init__(self, path):
        """
        constructor

        :param path: str; has to contain a valid path form containing images and labels
        """
        raise NotImplementedError("this is an abstract class; it exists just for interface reasons; you must instantiate a subclass.")


    def get_filelist(self):
        """
        returns a list of the files that this data loaded holds
        """
        raise NotImplementedError()


    def get_path(self):
        """
        returns the path from which the images are loaded
        """
        raise NotImplementedError()


    def get_next_batch(self):
        """
        returns the next batch of images & labels
        """
        raise NotImplementedError()


