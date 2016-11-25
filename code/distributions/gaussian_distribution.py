import numpy as np

class Gaussian:

    @staticmethod
    def get_normal_distribution(shape=(0, 10), mu=0.0, sigma=1.0):
        # discrete gaussian vector
        data_buffer = np.arange(shape[0], shape[1])

        data_buffer = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (data_buffer - mu) ** 2 / (2 * sigma ** 2))

        return data_buffer

    @staticmethod
    def get_noisy_normal_distribution(shape=(0, 10), mu=0.0, sigma=1.0, noise_amplifier=1.0):
        # creates a discrete gaussian vector with noise
        percentage = 0.15 * noise_amplifier  # maximum percentage to be added or substracted from original distribution

        distri = Gaussian.get_normal_distribution(shape, mu, sigma) 
        distri = distri * np.random.uniform(size=distri.shape, low=1.0-percentage, high=1.0+percentage)
        distri = distri + (np.random.uniform(size=distri.shape, low=-0.0005, high=0.0005) * (1 + (5.0 * noise_amplifier - 5.0)))

        return distri
