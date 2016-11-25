import sys
import os
import random
from itertools import izip

# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))




class Cycler(object):

    def __init__(self, data_object_list):
        self.__data_objects = data_object_list
        self.__data_object_loadings = dict() 
        self.__next_dataset = None
        self.__total_number_of_images = None
        self.__total_number_of_loadings = 0

        self.__build_data_object_dict(data_object_list)


    def get_all_data_objects(self):
        return self.__data_objects


    def get_next_dataset_for_batch_loading(self):
        self.__total_number_of_loadings += 1
        self.__choose_next_dataset()

        return self.__next_dataset
        

    def __build_data_object_dict(self, data_object_list):
        self.__total_number_of_images = 0
        for obj in data_object_list:
            self.__data_object_loadings[obj] = 0
            self.__total_number_of_images += obj.get_dataset_size()


    def __choose_next_dataset(self):
        # get a random number in range (1, total-number-of-images-in-all-datasets + 1)
        rnd_number = random.randrange(1, self.__total_number_of_images + 1)

        # initialize a counter
        counter = 0

        # iterate over all data objects
        for obj in self.__data_objects:
            # increase the counter by the number of images of the current dataset object
            counter += obj.get_dataset_size()

            # if the rnd_number is smaller than counter:
            # i.e. when   |        x    <=    y                        |  <-- range over all images
            #             ^        ^          ^                        ^
            #           start   rnd_number  counter                   end
            #       (first image)                                 (last image)
            if rnd_number <= counter:
                # select the current dataset for the next batch
                self.__next_dataset = obj

                # increase the number of loadings for the corresponding dataset object
                self.__data_object_loadings[obj] += 1

                # stop the loop -- randomly chosen dataset found
                break

    
    def get_loading_for_specific_dataset(self, obj):
        return self.__data_object_loadings[obj]


    def get_total_number_of_images(self):
        return self.__total_number_of_images

    
    def get_total_number_of_loadings(self):
        return self.__total_number_of_loadings



