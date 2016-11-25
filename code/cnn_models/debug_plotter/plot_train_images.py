from matplotlib import pyplot as plt
import os
import sys
import numpy as np

# append the experiments/code directory to python sys path
sys.path.append(str(os.environ["BTEX"]))

# network model
from cnn_models.models.cnn_s11_ex01 import NetworkModel as net_cls

# data object
from cnn_lib.BufferDataLoader import DataObject



# load dataset
tr_cnn_1 = DataObject(str(os.environ["BTDATA"]) + "cnn-training-1/test/", batch_size=1, xy_shape=(800, 600))

# network configuration
experiment = "cnn_exp10_01"
net = net_cls(experiment)
net.load_network()

# configuration
log_path = os.environ["BTLOG"] + experiment + "/" + "train_image_debug_plot/"
log_images = [1, 2, 3, 4, 7, 10, 18, 19, 25, 27, 28, 32, 35, 36, 45, 56, 66, 67, 68, 77]

# create or clear existing "train_image_debug_plot" directory
if os.path.exists(log_path) and os.path.isdir(log_path):
    # clear the directory for a new debug plot session
    for f in os.listdir(log_path):
        fpath = os.path.join(log_path, f)
        if os.path.isfile(fpath): 
            os.unlink(fpath)
else:
    os.makedirs(log_path)



# debug plot
for img_i in log_images:
    # get image data
    tr_imgdata, tr_Y_x, tr_Y_y, fname = tr_cnn_1.get_specific_image(img_i)
    # unpack data
    tr_imgdata = tr_imgdata[0]

    # run the network
    nn_out_x, nn_out_y, nn_out_x_softmax, nn_out_y_softmax, top11_x, top11_y = net.run_network([tr_imgdata])
    nn_out_x = nn_out_x[0]
    nn_out_y = nn_out_y[0]
    nn_out_x_softmax = nn_out_x_softmax[0]
    nn_out_y_softmax = nn_out_y_softmax[0]

    # generate heatmap for (x, y) - linear data
    heatmap_linear = np.zeros((600, 800))
    heatmap_linear += nn_out_x[None, :]
    heatmap_linear += nn_out_y[:, None]
    heatmap_linear *= 0.5

    # generate heatmap for (x, y) - softmax data
    heatmap_softmax = np.zeros((600, 800))
    heatmap_softmax += nn_out_x_softmax[None, :]
    heatmap_softmax += nn_out_y_softmax[:, None]
    heatmap_softmax *= 0.5

    # create the figure
    f = plt.figure()

    # create subplots
    # linear
    p_linear_img = plt.subplot2grid((2, 4), (0, 0), aspect="auto")
    p_linear_heatmap = plt.subplot2grid((2, 4), (0, 1), aspect="auto")
    p_linear_x = plt.subplot2grid((2, 4), (0, 2), aspect="auto")
    p_linear_y = plt.subplot2grid((2, 4), (0, 3), aspect="auto")
    # softmax
    p_softmax_img = plt.subplot2grid((2, 4), (1, 0), aspect="auto")
    p_softmax_heatmap = plt.subplot2grid((2, 4), (1, 1), aspect="auto")
    p_softmax_x = plt.subplot2grid((2, 4), (1, 2), aspect="auto")
    p_softmax_y = plt.subplot2grid((2, 4), (1, 3), aspect="auto")

    # plot the image
    p_linear_img.imshow(tr_imgdata, interpolation="None")
    p_softmax_img.imshow(tr_imgdata, interpolation="None")

    # plot the heatmaps
    p_linear_heatmap.imshow(tr_imgdata, interpolation="None")
    p_linear_heatmap.imshow(heatmap_linear, interpolation="None", alpha=0.35)
    p_softmax_heatmap.imshow(tr_imgdata, interpolation="None")
    p_softmax_heatmap.imshow(heatmap_softmax, interpolation="None", alpha=0.35)

    # plot linear data
    p_linear_x.plot(tr_Y_x, linestyle="solid", color="blue")
    p_linear_x.plot(nn_out_x, linestyle="solid", color="red")

    p_linear_y.plot(tr_Y_y, linestyle="solid", color="blue")
    p_linear_y.plot(nn_out_y, linestyle="solid", color="red")

    # plot softmax data
    p_softmax_x.plot(tr_Y_x, linestyle="solid", color="blue")
    p_softmax_x.plot(nn_out_x_softmax, linestyle="solid", color="red")

    p_softmax_y.plot(tr_Y_y, linestyle="solid", color="blue")
    p_softmax_y.plot(nn_out_y_softmax, linestyle="solid", color="red")

    # save the figure
    f.set_size_inches(21.0, 12.0)
    f.savefig(log_path + fname + ".png", dpi=80)
