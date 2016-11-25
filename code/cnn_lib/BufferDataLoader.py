from matplotlib import image as mpimg
from cnn_lib.AbstractDataLoader import AbstractDataObject
from distributions.gaussian_distribution import Gaussian as G 
from extra.img_ops import ImageOperations as ImgOps
from extra.array_type_printer import ArrayType as AT

from itertools import cycle
import numpy as np
import random
import copy
import os


class DataObject(AbstractDataObject):
    
    __files = None               # list of filenames (images)
    __dataset_size = None        # size of the dataset -> total amount of training/test files
    __path = None                # a valid path containing images and labels
    __labels = None              # a dictionary for the labels (str: img_filename) -> (x, y)
    __batch_size = None          # size of the image data batches that will be loaded
    __xy_shape = None            # tuple that contains the size of the x- and y-vector
    __inhibitor_dataset = None   # boolean flag -> inhibitor datasets have *no* labels; they just contain
                                 # negative sample data, therefore labels are always all zero
    __debug = None               # boolean flag -> en- or disables debug printing
    __sigma_rate = None          # normal distributions variance is set to:
                                 # -> vector shape (image width or height) / sigma_rate

    def __init__(self, path, batch_size=1, xy_shape=(800, 600), inhibitor_dataset=False,\
                 debug=False, sigma_rate=40.0):
        """
        constructor

        :param path: a valid path containing images and labels
        :type path: str

        :param batch_size: size of the image data batches that will be loaded
        :type batch_size: int
        """
        # init variables
        self.__files = list() 
        self.__labels = dict()
        self.__path = path 
        self.__batch_size = batch_size
        self.__inhibitor_dataset = inhibitor_dataset
        self.__debug = debug
        self.__xy_shape = xy_shape
        self.__file_cycler = None 
        self.__sigma_rate = sigma_rate
        self.__last_loaded_files = list()

        # check if the path is valid
        self.__check_path_validity()

        # traverse given directory and check for images
        # gives back a list of filenames into self.__files, if everything seems ok
        self.__files = sorted(self.__traverse_dir())
        self.__dataset_size = len(self.__files)

        # check if labels file is valid
        # builds up the dictionary self.__labels, if everything seems ok
        if not self.__inhibitor_dataset:
            self.__check_and_create_labels()

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
        # get image shape
        shape_x = self.__xy_shape[0]
        shape_y = self.__xy_shape[1]

        # get image filename
        fname = self.__files[i]

        # load the img data
        img_data = mpimg.imread(self.__path + fname + ".png")

        if img_data.shape[2] == 4:
            img_data = img_data[:, :, 0:3]

        # get the corresponding labels
        if self.__inhibitor_dataset:
            x, y = 0, 0
        else:
            x, y = self.__labels[fname]

        # distributions
        x_distri = G.get_normal_distribution(shape=(0, shape_x),
                                             mu=float(x),
                                             sigma=shape_x/self.__sigma_rate)

        y_distri = G.get_normal_distribution(shape=(0, shape_y),
                                             mu=float(y),
                                             sigma=shape_y/self.__sigma_rate)

        return np.array([img_data]), x_distri, y_distri, fname


    def get_next_batch(self):
        """
        returns the next batch of images & labels

        return format is 3-tupel of:
            [batch_size, img_data],
            [batch_size, x_gaussian_distribution]
            [batch_size, y_gaussian_distribution]

        return types are numpy ndarrays for all 3 tupel entries
        """
        # reset last loaded files list
        self.__last_loaded_files = list()

        buffer_img_data = list()
        buffer_x_distributions = list()
        buffer_y_distributions = list()

        shape_x = self.__xy_shape[0]
        shape_y = self.__xy_shape[1]

        # create the next batch
        for i in range(self.__batch_size):
            # get the next element out of the iterator
            fname = self.__file_cycler.next()

            self.__last_loaded_files.append(fname)

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
                buffer_x_distributions.append(np.zeros(self.__xy_shape[0], dtype=np.float32))
                buffer_y_distributions.append(np.zeros(self.__xy_shape[1], dtype=np.float32))
            else:
                # get the corresponding labels
                x, y = self.__labels[fname]

                # insert the full gaussian distribution for x data
                buffer_x_distributions.append(
                        G.get_normal_distribution(shape=(0, shape_x),
                                                         mu=float(x),
                                                         sigma=shape_x/40.0)
                        )

                # insert the full gaussian distribution for x data
                buffer_y_distributions.append(
                        G.get_normal_distribution(shape=(0, shape_y),
                                                         mu=float(y),
                                                         sigma=shape_y/40.0)
                        )

        # return the complete/full arrays for training
        return np.array(buffer_img_data).astype(np.float32),\
               np.array(buffer_x_distributions).astype(np.float32),\
               np.array(buffer_y_distributions).astype(np.float32)


    def get_last_loaded_files(self):
        return self.__last_loaded_files


    def debug_print(self):
        if not self.__debug:
            raise Exception("the debug flag was not set; no debug information available!")

        print("\nloadings of every file:")
        for key, val in self.__file_loadcount.iteritems():
            print(key, val)


    def get_data_dict(self):
        if self.__data_rdy_to_load:
            size = len(self.__files)  # get number of images
            container = []  # container for image date

            #######################
            # traverse all images #
            #######################
            for img_str in self.__files:
                container.append(mpimg.imread(self.__path + img_str))  # append image data to container

            self.__data_dict["img_data"] = np.array(container)  # set data, convert it to numpy array

            ###################
            # build up labels #
            ###################
            with open(self.__path + "labels.txt", "r") as f:  # read file containing labels
                label_buffer = f.readlines()  # read in all lines

            # filter all lines not containing "===" <-- separator for img label data
            label_buffer = filter(lambda elem: "===" in elem, label_buffer)

            # remove linebreaks
            label_buffer = [elem.replace("\n", "") for elem in label_buffer]

            # sort labels
            label_buffer = sorted(label_buffer)

            # split file names and label data
            label_buffer = [elem.split("===") for elem in label_buffer]

            label_container_x = []
            label_container_y = []

            # iterate over all files
            for img_str in self.__files:

                # Flag for determining if a label for any image is missing
                label_found = False

                # iterate over the label buffer
                for elem in label_buffer:

                    # search for the correct label
                    if elem[0] in img_str:
                        # the correct label was found
                        label_found = True

                        # extract x, y coordinates
                        x, y = elem[1].split(",")

                        # coordinates are formatted like e.g. x256, y461
                        x = int(x[1:])  # deletes the first character, just leaves the integer
                        y = int(y[1:])  # deletes the first character, just leaves the integer

                        # append data to the container
                        label_container_x.append(np.array([x]).astype(np.int32))
                        label_container_y.append(np.array([y]).astype(np.int32))

                # if the correct label was NOT found
                if not label_found:
                    exit("Error: Missing label. The label for the image {} was not found.".format(img_str))

            self.__data_dict["img_labels_x"] = np.array(label_container_x)
            self.__data_dict["img_labels_y"] = np.array(label_container_y)

            return self.__data_dict

        else:
            exit("Error: data (a directory) must be loaded before data extraction can start.")


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


    def __check_and_create_labels(self):
        """
        checks the labels file
        """
        # check that the path exists
        if not os.path.exists(self.__path + "labels.txt"):
            raise IOError("labels file '{}' does not exist".format(self.__path + "labels.txt"))

        # check that the labels file exists
        if not os.path.isfile(self.__path + "labels.txt"):
            raise IOError("'{}' is not a file".format(self.__path + "labels.txt"))

        # fill the self.__labels dictionary
        self.__labels = self.__read_label_file()

        # check if label data for every image is available
        for f in self.__files:
            if not self.__labels.has_key(f):
                raise Exception("the label for image '{}' is missing".format(f))


    def __read_label_file(self):
        """
        reads in the contents of the labels file
        """
        # load contents of labels file        
        label_buffer = []

        # read the file's contents
        with open(self.__path + "labels.txt", "r") as f_labels:  # read file containing labels
            label_buffer = f_labels.readlines()  # read in all lines

        # filter all lines not containing "===" <-- separator for img label data
        label_buffer = filter(lambda e: "===" in e, label_buffer)

        # remove linebreaks and potential whitespaces
        label_buffer = [e.strip() for e in label_buffer]

        # create the dictionary
        labels = dict()

        for e in label_buffer:
            # seperate img filename and xy label data
            fname, xydata = e.split("===")

            # seperate x and y label data
            x, y = xydata.split(",")

            # add the entry to the dictionary 
            # (str: filename) -> (tuple (int: x, int: y))
            # caution: first char in x/y is "x"/"y"!
            # original encoding looks like this: x123,y456
            labels[fname] = (int(x[1:]), int(y[1:]))

        return labels


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
