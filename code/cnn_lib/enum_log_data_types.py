from enum import Enum
from collections import namedtuple
from enum_colors import COLORS


data_settings = namedtuple("data_settings", ["description", "label", "color", "marker", "linestyle"])


class LOG_DATA_TYPES(Enum):
    """
    enum for the log data
    """
    #####################
    # original img data #
    #####################
    o_img_data = "original image data"

    ##################
    # network output #
    ##################
    nn_out_x = data_settings(description="x output of the network",
                             label="x_out",
                             color=COLORS.dark_purple,
                             marker="x",
                             linestyle="None")

    nn_out_y = data_settings(description="y output of the network",
                             label="y_out",
                             color=COLORS.dark_purple,
                             marker="x",
                             linestyle="None")

    ###################
    # training singal #
    ###################
    nn_tr_x = data_settings(description="training data for x output",
                            label="x_train",
                            color=COLORS.black,
                            marker="x",
                            linestyle="None")

    nn_tr_y = data_settings(description="training data for y output",
                            label="y_train",
                            color=COLORS.black,
                            marker="x",
                            linestyle="None")

    ###################
    # test error data #
    ###################
    err_test_x = data_settings(description="x error for test images",
                               label="err x",
                               color=COLORS.red,
                               marker="x",
                               linestyle="solid")

    err_test_y = data_settings(description="y error for test images",
                               label="err y",
                               color=COLORS.blue,
                               marker="x",
                               linestyle="solid")

    err_test_total = data_settings(description="total (x/y combined) error for test images",
                                   label="err total",
                                   color=COLORS.grey,
                                   marker="x",
                                   linestyle="solid")

    ################
    # top-11 error #
    ################

    top11_test_x = data_settings(description="x top-11 error for test images",
                                 label="top11 test x",
                                 color=COLORS.green,
                                 marker="x",
                                 linestyle="solid")

    top11_test_y = data_settings(description="y top-11 error for test images",
                                 label="top11 test y",
                                 color=COLORS.orange,
                                 marker="x",
                                 linestyle="solid")

    top11_train_x = data_settings(description="x top-11 error for train images",
                                  label="top11 train x",
                                  color=COLORS.dark_green,
                                  marker="x",
                                  linestyle="solid")

    top11_train_y = data_settings(description="y top-11 error for train images",
                                  label="top11 train y",
                                  color=COLORS.dark_orange,
                                  marker="x",
                                  linestyle="solid")

    #################
    # accuracy data #
    #################
    accu_x = data_settings(description="accuracy output for x on test images",
                           label="acc x",
                           color=COLORS.dark_red,
                           marker="x",
                           linestyle="solid")

    accu_y = data_settings(description="accuracy output for y on test images",
                           label="acc y",
                           color=COLORS.dark_blue,
                           marker="x",
                           linestyle="solid")

    accu_total = data_settings(description="accuracy output for x/y combined on test images",
                               label="acc total",
                               color=COLORS.black,
                               marker="x",
                               linestyle="solid")

    ######################
    # cross entropy data #
    ######################
    cross_entropy_x = data_settings(description="modelling the x cross entropy error",
                                    label="ce x",
                                    color=COLORS.dark_red,
                                    marker="x",
                                    linestyle="solid")

    cross_entropy_y = data_settings(description="modelling the y cross entropy error",
                                    label="ce y",
                                    color=COLORS.dark_blue,
                                    marker="x",
                                    linestyle="solid")

    ######################
    # cost function data #
    ######################
    cost_function_x = data_settings(description="total x error, calculated by the networks cost function",
                                    label="err x",
                                    color=COLORS.red,
                                    marker="x",
                                    linestyle="solid")

    cost_function_y = data_settings(description="total y error, calculated by the networks cost function",
                                    label="err y",
                                    color=COLORS.blue,
                                    marker="x",
                                    linestyle="solid")

    ############################
    # integral constraint data #
    ############################
    integral_constraint_x = data_settings(description="integral constraint x 'error', models the loss of the total sum",
                                          label="int x",
                                          color=COLORS.dark_lime,
                                          marker="x",
                                          linestyle="solid")
    integral_constraint_y = data_settings(description="integral constraint y 'error', models the loss of the total sum",
                                          label="int y",
                                          color=COLORS.orange,
                                          marker="x",
                                          linestyle="solid")
