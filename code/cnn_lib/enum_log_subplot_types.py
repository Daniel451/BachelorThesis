from enum import Enum
from collections import namedtuple


plot_type_entry = namedtuple("plot_type_entry", ["description", "label"])


class LOG_SUBPLOT_TYPES(Enum):
    """
    enum for plot types like 'image', 'cost_function', 'network_output', ...
    """
    o_img = plot_type_entry(description="original image", label="original image")
    nn_x = plot_type_entry(description="output of the network for the ball's x-coordinate", label="ball x")
    nn_y = plot_type_entry(description="output of the network for the ball's y-coordinate", label="ball y")
    cost = plot_type_entry(description="cost function plot", label="cost function")
    test_err = plot_type_entry(description="test error plot", label="evaluation")

