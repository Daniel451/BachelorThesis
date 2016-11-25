import sys
import os
import random
from itertools import izip

# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))

from cnn_adv_lib.DataObjectCycler import Cycler as DOC
from cnn_lib.BufferDataLoader import DataObject as DataObject


# create dataset objects
tr_cnn_1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_1_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_3 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-3-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_3_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-3-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_leipzig1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig1-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_leipzig1_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig1-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_leipzig2 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig2-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_leipzig2_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig2-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_leipzig3 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig3-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_leipzig3_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig3-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_leipzig4 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig4-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_leipzig4_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig4-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))


# stack data objects in a list
training_data = [tr_cnn_1, tr_cnn_1_mirrored,\
                 tr_cnn_3, tr_cnn_3_mirrored,\
                 tr_cnn_leipzig1, tr_cnn_leipzig1_mirrored,\
                 tr_cnn_leipzig2, tr_cnn_leipzig2_mirrored,\
                 tr_cnn_leipzig3, tr_cnn_leipzig3_mirrored,\
                 tr_cnn_leipzig4, tr_cnn_leipzig4_mirrored]


# create a new data object cycler
do_cycler = DOC(training_data)


total_number_of_loadings = int(1e6)
for i in xrange(1, total_number_of_loadings + 1, 1):
    do_cycler.get_next_dataset_for_batch_loading()


def print_loadings(obj_name, obj):
    loadings = do_cycler.get_loading_for_specific_dataset(obj)
    percent_of_images = (100.0 / float(do_cycler.get_total_number_of_images())) * float(obj.get_dataset_size())
    percent_of_loadings = (100.0 / float(do_cycler.get_total_number_of_loadings())) * float(loadings)
    
    print("---> [{:>6,} images | loaded: {:>7,} times | % of total images ->{: >6.2f}% <=>{: >6.2f}% <- % of total loadings] in dataset"\
          .format(obj.get_dataset_size(), loadings, percent_of_images, percent_of_loadings)\
          + ("'{}'".format(obj_name)).rjust(30, "."))

print("######################################################################################")
print("-> total number of loadings: {:,}".format(total_number_of_loadings))
print("-> total number of loadings according to DOC: {:,}".format(do_cycler.get_total_number_of_loadings()))
print("-")
print_loadings("tr_cnn_1", tr_cnn_1)
print_loadings("tr_cnn_1_mirrored", tr_cnn_1_mirrored)
print_loadings("tr_cnn_3", tr_cnn_3)
print_loadings("tr_cnn_3_mirrored", tr_cnn_3_mirrored)
print_loadings("tr_cnn_leipzig1", tr_cnn_leipzig1)
print_loadings("tr_cnn_leipzig1_mirrored", tr_cnn_leipzig1_mirrored)
print_loadings("tr_cnn_leipzig2", tr_cnn_leipzig2)
print_loadings("tr_cnn_leipzig2_mirrored", tr_cnn_leipzig2_mirrored)
print_loadings("tr_cnn_leipzig3", tr_cnn_leipzig3)
print_loadings("tr_cnn_leipzig3_mirrored", tr_cnn_leipzig3_mirrored)
print_loadings("tr_cnn_leipzig4", tr_cnn_leipzig4)
print_loadings("tr_cnn_leipzig4_mirrored", tr_cnn_leipzig4_mirrored)





