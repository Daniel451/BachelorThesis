import sys
import os
import numpy as np
from itertools import izip, cycle
from matplotlib import pyplot as plt
from matplotlib import gridspec

# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))

from cnn_lib.BufferDataLoader import DataObject
from extra.array_type_printer import ArrayType as AT


print("creating data objects...")
# create data objects
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



#te_cnn_1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-small/test/", batch_size=1, xy_shape=(200, 150))
print("finished creating data objects\n")


#training_data = [tr_cnn_1, tr_cnn_1_mirrored, tr_cnn_3, tr_cnn_3_mirrored, tr_cnn_leipzig1, tr_cnn_leipzig1_mirrored]
#training_data = [tr_cnn_leipzig2]
#training_data = [tr_cnn_leipzig2, tr_cnn_leipzig2_mirrored, tr_cnn_leipzig3, tr_cnn_leipzig3_mirrored, tr_cnn_leipzig4, tr_cnn_leipzig4_mirrored]
training_data = [tr_cnn_1, tr_cnn_1_mirrored, tr_cnn_3, tr_cnn_3_mirrored, tr_cnn_leipzig2, tr_cnn_leipzig2_mirrored, tr_cnn_leipzig3, tr_cnn_leipzig3_mirrored, tr_cnn_leipzig4, tr_cnn_leipzig4_mirrored]

f = plt.figure()
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
o_img = plt.subplot(gs[0, 0])
x = plt.subplot(gs[1, 0])
y = plt.subplot(gs[0, 1])
y.invert_xaxis()
y.invert_yaxis()
img = plt.subplot(gs[1, 1])

data_iter = cycle(training_data)

title_fontsize = 32

for i in xrange(100):
    next_data_object = data_iter.next()

    imgdata, xdistri, ydistri = next_data_object.get_next_batch()
    fnames = next_data_object.get_last_loaded_files()
    AT.print_type(imgdata)

    xmax = np.argmax(xdistri[0])
    ymax = np.argmax(ydistri[0])
    AT.print_type(xdistri)
    AT.print_type(ydistri)

    o_img.cla()
    img.cla()
    x.cla()
    y.cla()

    o_img.imshow(np.copy(imgdata[0]))
    o_img.set_title("original image\n'{}'".format(fnames[0]), fontsize=title_fontsize)

    # draw a black cross
    imgdata[0][ymax-1:ymax+2, xmax-10:xmax+11, :] = [0.0, 0.0, 0.0]
    imgdata[0][ymax-10:ymax+11, xmax-1:xmax+2, :] = [0.0, 0.0, 0.0]

    # draw a red cross
    imgdata[0][ymax, xmax-10:xmax+11, :] = [1.0, 0.0, 0.0]
    imgdata[0][ymax-10:ymax+11, xmax, :] = [1.0, 0.0, 0.0]

    img.imshow(imgdata[0])
    img.set_title("image label plot\nx={} | y={}".format(fnames[0], xmax, ymax), fontsize=title_fontsize)

    x.plot(np.arange(xdistri[0].size), xdistri[0])
    x.set_title("x={}".format(xmax), fontsize=title_fontsize)

    y.plot(ydistri[0], np.arange(ydistri[0].size))
    y.set_title("y={}".format(ymax), fontsize=title_fontsize)

    plt.draw()
    f.ginput(n=1, timeout=0)








