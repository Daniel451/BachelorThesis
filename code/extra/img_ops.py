import numpy as np

class ImageOperations(object):

    @staticmethod
    def rescale_values(oimg, upper_bound=1.0, dtype=np.float32):
        """
        rescales the values in an image array to match the interval [0, upper_bound]

        :param oimg: original image; only a copy is altered
        :param upper_bound: upper bound that should not be exceeded by the values in an image array
        :param dtype: data type; defaults to np.float32
        """
        img = np.copy(oimg)
        img = img.astype(np.float64)
        img *= upper_bound / np.max(img)
        img = img.clip(0.0, upper_bound)

        return img.astype(dtype)
        



