import numpy as np

class ArrayType:

    @staticmethod
    def get_type(arr):
        if type(arr) == np.ndarray:
            return "Type is: {} | shape: {} | dtype: {} | max: {} | min: {} | mean: {} | median: {}"\
                .format(type(arr), arr.shape, arr.dtype, np.max(arr), np.min(arr), np.mean(arr), np.median(arr))
        else:
            return "Type is: {}".format(type(arr))

    @staticmethod
    def print_type(arr):
        print(ArrayType.get_type(arr))
