import sys
import os
# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))
from cnn_lib.BufferDataLoader import DataObject
from extra.array_type_printer import ArrayType as AT
from itertools import izip
import numpy as np

from matplotlib import pyplot as plt

D = DataObject("/home/daniel/Dropbox/Studium/computer science BSc/bachelor-thesis/bt-img/cnn-training-1/train/", batch_size=1, debug=True)


imgdata, trX, trY = D.get_next_batch()

AT.print_type(imgdata)
AT.print_type(trX)
AT.print_type(trY)

for x, y in izip(trX, trY):
    print(np.argmax(x), np.argmax(y))

# plot and check one single image
pimg = plt.subplot2grid((1, 3), (0, 0))
px = plt.subplot2grid((1, 3), (0, 1))
py = plt.subplot2grid((1, 3), (0, 2))

pimg.imshow(imgdata[0])
px.plot(trX[0])
px.set_title("x = {}".format(np.argmax(trX[0])))
py.plot(trY[0])
py.set_title("y = {}".format(np.argmax(trY[0])))
plt.show()


for e in [1, 5, 10, 1080]:
    pimg = plt.subplot2grid((1, 3), (0, 0))
    px = plt.subplot2grid((1, 3), (0, 1))
    py = plt.subplot2grid((1, 3), (0, 2))

    l_imgdata, l_xdistri, l_ydistri, l_fname = D.get_specific_image(e)
    print(l_fname)
    pimg.imshow(l_imgdata[0])
    px.plot(l_xdistri)
    px.set_title("x = {}".format(np.argmax(l_xdistri)))
    py.plot(l_ydistri)
    py.set_title("y = {}".format(np.argmax(l_ydistri)))
    plt.show(True)



# check loading of a specific file

# check loadings of every file
#for i in xrange(10):
    #D.get_next_batch()

#D.debug_print()
