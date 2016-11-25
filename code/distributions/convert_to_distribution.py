from gaussian_distribution import Gaussian
import numpy as np


class DistributionConverter:

    @staticmethod
    def convert_to_normal_distribution(arr, distribution_length):
        container = []

        for item in arr:
            container.append(
                    np.array(
                            Gaussian.get_normal_distribution(shape=(0, distribution_length), mu=item, sigma=20.0)
                    )
            )

        return np.array(container).astype(np.float32)
