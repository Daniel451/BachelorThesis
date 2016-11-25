import sys
import os
sys.path.append(str(os.environ["BTEX"]))

from cnn_lib.BufferDataLoaderWeaklySupervised import DataObject 
from extra.array_type_printer import ArrayType as AT
from matplotlib import pyplot as plt


test_data_neg = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-small/train/", batch_size=1, inhibitor_dataset=True)
test_data_pos = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-small/train/", batch_size=1, inhibitor_dataset=False)


number_of_test = 5

for i in xrange(number_of_test):
    neg_img, neg_label = test_data_neg.get_next_batch()
    
    plt.clf()
    plt.imshow(neg_img[0])
    plt.title("img {} of {} | label: {}".format(i+1, number_of_test, neg_label))
    plt.show()


for i in xrange(5):
    pos_img, pos_label = test_data_pos.get_next_batch()

    plt.clf()
    plt.imshow(pos_img[0])
    plt.title("img {} of {} | label: {}".format(i+1, number_of_test, pos_label))
    plt.show()


