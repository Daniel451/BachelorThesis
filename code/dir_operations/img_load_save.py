from matplotlib import image as mpimg
import numpy as np
import os


class ImageLoadSave:
    def __init__(self):
        self.__files = []
        self.__path = ""
        self.__data_dict = dict()
        self.__data_rdy_to_load = False


    def get_filelist(self):
        return self.__files


    def get_path(self):
        return self.__path


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


    def load_dir(self, read_dir):
        # check if directory exists
        if not os.path.exists(read_dir):
            exit("Error: Path does not exist.")

        # check if directory is valid directory
        if not os.path.isdir(read_dir):
            exit("Error: This is not a valid directory.")

        # traverse the directory and get a list of files
        self.__files = sorted(self.__traverse_dir(read_dir))

        # save path
        self.__path = read_dir

        # set flag data_rdy_to_load for enabling data loading
        self.__data_rdy_to_load = True


    def filter_exclude(self, ex_str):
        # excludes any files where ex_str can be found in the filename
        self.__files = filter(lambda item: ex_str not in item, self.__files)


    def __traverse_dir(self, read_dir):
        # get contents of directory
        contents = os.listdir(read_dir)

        # filter - exclude all but .png files
        contents = filter(lambda item: item[-4:] == ".png", contents)

        # check files are present
        if not len(contents) > 0:
            exit("Error: no images present in given directory.")

        return contents
