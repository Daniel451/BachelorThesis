#!/sw/env/gcc-4.9.3_openmpi-1.8.8/pkgsrc/2015Q4/bin/python2

import sys
import os
from itertools import izip
from matplotlib import pyplot as plt
from matplotlib import gridspec

# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))

from cnn_lib.BufferDataLoader import DataObject
from extra.array_type_printer import ArrayType as AT
from cnn_models.models.cnn_s11_ex15 import NetworkModel as net_cls


print("creating data objects...")
# create data objects
tr_cnn_1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_1_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_3 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-3-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_3_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-3-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

tr_cnn_leipzig1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig1-small/train/", batch_size=1, xy_shape=(200, 150))
tr_cnn_leipzig1_mirrored = DataObject(str(os.environ["BTDATA"]) + "cnn-training-leipzig1-mirrored-small/train/", batch_size=1, xy_shape=(200, 150))

te_cnn_1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1-small/test/", batch_size=1, xy_shape=(200, 150))
print("finished creating data objects\n")
sys.stdout.flush()

# stack data objects in a list
training_data = [tr_cnn_1, tr_cnn_1_mirrored, tr_cnn_3, tr_cnn_3_mirrored, tr_cnn_leipzig1, tr_cnn_leipzig1_mirrored]



# create the net and start training
net = net_cls("cnn_exp11_15")
net.train(training_data, te_cnn_1, log_every_n_steps=500, amount_of_train_steps=int(2e5), dropout_rate=1.0)




print("\nscript finished; normal termination\n")
sys.stdout.flush()

