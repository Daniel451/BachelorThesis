import cv2
import numpy as np
import time
import numpy as np
from extra.img_ops import ImageOperations as ImgOps
from matplotlib import pyplot as plt

def net_api(net, img):
    img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
    img_net = ImgOps.rescale_values(img, upper_bound=1.0, dtype=np.float32)
    
    print("start processing...")
    time_start = time.time()
    nn_out_x, nn_out_y, nn_out_x_softmax, nn_out_y_softmax, top11_x, top11_y = net.run_network([img_net])
    print("finished processing! Execution time: {:.4f}".format(time.time()-time_start))

    # unpack values
    nn_out_x = nn_out_x[0]
    nn_out_y = nn_out_y[0]
    nn_out_x_softmax = nn_out_x_softmax[0]
    nn_out_y_softmax = nn_out_y_softmax[0]

    # generate heatmap for (x, y)
    heatmap = np.zeros((600, 800))
    heatmap += nn_out_x_softmax[None, :]
    heatmap += nn_out_y_softmax[:, None]
    heatmap *= 0.5
    #heatmap = ImgOps.rescale_values(heatmap, upper_bound=255.0, dtype=np.uint8)

    heatmap_linear = np.zeros((600, 800))
    heatmap_linear += nn_out_x[None, :]
    heatmap_linear += nn_out_y[:, None]
    heatmap_linear *= 0.5

    #out_heatmap = cv2.addWeighted(img_grey, 0.5, heatmap, 0.8, 1.0) 
    #out_heatmap = cv2.applyColorMap(out_heatmap, cv2.COLORMAP_JET)

    p_img = plt.subplot2grid((2, 4), (0, 0), aspect="auto")
    p_heatmap = plt.subplot2grid((2, 4), (0, 1), aspect="auto")
    p_x = plt.subplot2grid((2, 4), (0, 2), aspect="auto")
    p_y = plt.subplot2grid((2, 4), (0, 3), aspect="auto")

    pl_img = plt.subplot2grid((2, 4), (1, 0), aspect="auto")
    pl_heatmap = plt.subplot2grid((2, 4), (1, 1), aspect="auto")
    pl_x = plt.subplot2grid((2, 4), (1, 2), aspect="auto")
    pl_y = plt.subplot2grid((2, 4), (1, 3), aspect="auto")

    #cv2.imshow("network heatmap", out_heatmap)
    #cv2.imshow("network x=" + np.argmax(nn_out_x_softmax), nn_out_x_softmax)
    #cv2.imshow("network y=" + np.argmax(nn_out_y_softmax), nn_out_y_softmax)

    # softmax
    p_img.imshow(img, interpolation="None")

    p_heatmap.imshow(img, interpolation="None")
    p_heatmap.imshow(heatmap, interpolation="None", alpha=0.35)

    p_x.plot(nn_out_x_softmax)
    p_x.set_title("x={}".format(np.argmax(nn_out_x_softmax)))
    p_y.plot(nn_out_y_softmax)
    p_y.set_title("y={}".format(np.argmax(nn_out_y_softmax)))

    # linear
    pl_img.imshow(img, interpolation="None")

    pl_heatmap.imshow(img, interpolation="None")
    pl_heatmap.imshow(heatmap_linear, interpolation="None", alpha=0.35)

    pl_x.plot(nn_out_x)
    pl_x.set_title("x={}".format(np.argmax(nn_out_x)))
    pl_y.plot(nn_out_y)
    pl_y.set_title("y={}".format(np.argmax(nn_out_y)))

    plt.show()


