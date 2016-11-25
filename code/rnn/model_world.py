import numpy as np
import matplotlib.pyplot as plt
import random
from functools import wraps
from gaussian_distribution import Gaussian


class World():
    """
    this class models the world for automatic data generation to (pre)train the RNN
    """

    sensor_last = None              # last sensor vector
    sensor_current = None           # current sensor vector
    sensor_last_noise = None        # last sensor vector - with noise
    sensor_current_noise = None     # current sensor vector - with noise
    sensor_count = None             # counts the amount of updates since last re-init
    distri_direction = None         # last object position
    distri_pos = None               # current object position
    distri_sigma = None             # sigma of the normal distribution
    distri_size = None              # size of the normal distribution (vector size)
    distri_step_size = None         # step size for distribution movement

    def __init__(self):
        self.distri_sigma = 20.0
        self.distri_size = (0, 800)
        self.distri_step_size = 40.0
        self.distri_pos = random.randint(self.distri_size[0], self.distri_size[1])

        self.sensor_current = self.__get_new_distribution()
        self.sensor_last = self.sensor_current

        self.sensor_current_noise = self.__get_new_distribution_noise()
        self.sensor_last_noise = self.sensor_current_noise

        self.sensor_count = 0

        self.distri_direction = random.randint(0, 1)


    def update_world(self):
        # update the sensor
        self.__sensor_update()    

        # create a new distribution
        self.sensor_current = self.__get_new_distribution()
        self.sensor_current_noise = self.__get_new_distribution_noise()


    def __sensor_update(self):
        # copy current data vector to sensor_last
        self.sensor_last = self.sensor_current
        self.sensor_last_noise = self.sensor_current_noise

        # move the distribution
        self.__move_distri_position()


    def get_current_sensor_data(self):
        return self.sensor_current


    def get_last_sensor_data(self):
        return self.sensor_last


    def get_current_sensor_data_noise(self):
        return self.sensor_current_noise


    def get_last_sensor_data_noise(self):
        return self.sensor_last_noise


    def __move_distri_position(self):
        # update sensor count since last re-init
        self.sensor_count += 1

        # change direction with a 1% chance
        if random.randint(1, 100) > 99:
            # direction should be changed
            if self.distri_direction == 0:
                self.distri_direction = 1
            else:
                self.distri_direction = 0

        # move distribution to the left or right
        # 0 = left, 1 = right
        if self.distri_direction == 0:
            self.distri_pos -= self.distri_step_size
        else:
            self.distri_pos += self.distri_step_size

        # if left or right border is passed, restart with new random position
        # and initialize a new speed (distri step size)
        if self.distri_pos < self.distri_size[0] or self.distri_pos > self.distri_size[1]:
            self.__new_init()

        # if the stepsize is very small, it could take a very long time
        # for the distribution to reach a the left or right border
        # therefore it should be reset after passing a certain threshold
        if self.sensor_count >= 30:
            self.__new_init()

        # in the real world, due to friction, an object normally slows down
        # over time, thus this should decrease movement over time
        self.__reduce_speed()


    def __reduce_speed(self):
        self.distri_step_size = self.distri_step_size * \
                                (1 - (-0.0189562 * np.log(0.0019808 * (self.distri_step_size + 1e-6))))
        #print("step size:         {: >6.3f}".format(self.distri_step_size))
        #print("sensor count:      {:6}".format(self.sensor_count))


    def __new_init(self):
        self.sensor_count = 0
        self.distri_pos = random.randint(self.distri_size[0], self.distri_size[1])
        self.distri_step_size = float(random.randint(10, 90))


    
    def __get_new_distribution(self):
        return np.array([Gaussian.get_normal_distribution(shape=self.distri_size, \
                        mu=self.distri_pos, \
                        sigma=self.distri_sigma)])


    def __get_new_distribution_noise(self):
        return np.array([Gaussian.get_noisy_normal_distribution(shape=self.distri_size, \
                        mu=self.distri_pos, \
                        sigma=self.distri_sigma)])


    def plot(self):
        plt.clf()
        plt.plot(self.get_current_sensor_data()[0], color="blue", marker="None", linestyle="solid")
        plt.plot(self.get_last_sensor_data()[0], color="red", marker="None", linestyle="solid")
        plt.draw()
        plt.pause(0.25)


    def plot_single_frame(self):
        plt.clf()
        plt.plot(self.get_current_sensor_data()[0], color="blue", marker="None", linestyle="solid")
        plt.plot(self.get_last_sensor_data()[0], color="red", marker="None", linestyle="solid")
        plt.show()





