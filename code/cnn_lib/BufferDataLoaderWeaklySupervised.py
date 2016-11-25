import sys
import os
sys.path.append(str(os.environ["BTEX"]))
from extra.img_ops import ImageOperations as ImgOps
from extra.array_type_printer import ArrayType as AT

from matplotlib import image as mpimg
from itertools import cycle
import numpy as np
import random
import copy


class DataObject(object):
    
    __files = None               # list of filenames (images)
    __dataset_size = None        # size of the dataset -> total amount of training/test files
    __path = None                # a valid path containing images and labels
    __labels = None              # a dictionary for the labels (str: img_filename) -> (x, y)
    __batch_size = None          # size of the image data batches that will be loaded
    __inhibitor_dataset = None   # boolean flag -> inhibitor datasets have *no* labels; they just contain
                                 # negative sample data, therefore labels are always all zero
    __debug = None               # boolean flag -> en- or disables debug printing

    def __init__(self, path, batch_size=1, inhibitor_dataset=False,\
                 debug=False):
        """
        constructor

        :param path: a valid path containing images and labels
        :type path: str

        :param batch_size: size of the image data batches that will be loaded
        :type batch_size: int
        """
        # init variables
        self.__files = []
        self.__labels = dict()
        self.__path = path 
        self.__batch_size = batch_size
        self.__inhibitor_dataset = inhibitor_dataset
        self.__debug = debug
        self.__file_cycler = None 

        # check if the path is valid
        self.__check_path_validity()

        # traverse given directory and check for images
        # gives back a list of filenames into self.__files, if everything seems ok
        self.__files = sorted(self.__traverse_dir())
        self.__dataset_size = len(self.__files)

        # prepare for batch loading
        # creates a random copy of filenames and a cycle iterator over it
        # self.__file_cycler is created after method finishes
        self.__prepare_batch_loading()

        # prepare debugging
        if self.__debug:
            self.__prepare_debugging()


    def __prepare_debugging(self):
        # create a dictionary to log the number of
        # (get_next_batch) loads per image
        # dict: filename -> number-of-loadings
        self.__file_loadcount = dict()

        # create zero entries for each filename
        for e in self.__files:
            self.__file_loadcount[e] = 0


    def __prepare_batch_loading(self):
        rnd_lst = copy.copy(self.__files)
        random.shuffle(rnd_lst)

        self.__file_cycler = cycle(rnd_lst)



    def get_filelist(self):
        """
        returns a list of the files that this data loaded holds
        """
        return self.__files


    def get_dataset_size(self):
        """
        returns the size of the dataset -> total number of files
        """
        return self.__dataset_size


    def get_path(self):
        """
        returns the path from which the images are loaded
        """
        return self.__path


    def get_specific_image(self, i):
        """
        returns one numpy array of shape (1, height, width, 3) for image i
        -> img_array, label_distribution_x, label_distribution_y, filename

        :param i: int -> index of the image
        """
        # get image filename
        fname = self.__files[i]

        # load the img data
        img_data = mpimg.imread(self.__path + fname + ".png")

        if img_data.shape[2] == 4:
            img_data = img_data[:, :, 0:3]

        # get the corresponding labels
        if self.__inhibitor_dataset:
            ball_present = 0.0 
        else:
            ball_present = 1.0

        return np.array([img_data]), np.array([[ball_present]]), fname


    def get_next_batch(self):
        """
        returns the next batch of images & labels

        return format is 3-tupel of:
            [batch_size, img_data],
            [batch_size, x_gaussian_distribution]
            [batch_size, y_gaussian_distribution]

        return types are numpy ndarrays for all 3 tupel entries
        """
        buffer_img_data = list()
        buffer_ball_present = list()

        # create the next batch
        for i in range(self.__batch_size):
            # get the next element out of the iterator
            fname = self.__file_cycler.next()

            # load the img data
            img_data = mpimg.imread(self.__path + fname + ".png")

            if img_data.shape[2] == 4:
                img_data = img_data[:, :, 0:3]

            # log this load
            if self.__debug:
                self.__file_loadcount[fname] += 1

            # insert img_data into the buffer
            buffer_img_data.append(img_data)

            if self.__inhibitor_dataset:
                buffer_ball_present.append([0.0])
            else:
                buffer_ball_present.append([1.0])

        # return the complete/full arrays for training
        return np.array(buffer_img_data).astype(np.float32),\
               np.array(buffer_ball_present).astype(np.float32)


    def debug_print(self):
        if not self.__debug:
            raise Exception("the debug flag was not set; no debug information available!")

        print("\nloadings of every file:")
        for key, val in self.__file_loadcount.iteritems():
            print(key, val)


    def __check_path_validity(self):
        # check if path is a str
        if not type(self.__path) == str:
            raise TypeError("path has to be a string but was {}".format(type(self.__path)))

        # check if directory exists
        if not os.path.exists(self.__path):
            raise IOError("specified path '{}' does not exist".format(self.__path))

        # check if directory is valid directory
        if not os.path.isdir(self.__path):
            raise IOError("specified path '{}' is not a valid directory".format(self.__path))


    def __traverse_dir(self):
        """
        traverses the given directory and gives back all image files
        """
        # get contents of directory
        contents = os.listdir(self.__path)

        # filter - exclude all but .png files
        contents = filter(lambda item: item[-4:] == ".png", contents)

        # check files are present
        if not len(contents) > 0:
            raise Exception("no matching images (png-files) present in given directory '{}'".format(self.__path))

        # deletes the file extension '.png' for every entry
        contents = [e[:-4] for e in contents]

        return contents
