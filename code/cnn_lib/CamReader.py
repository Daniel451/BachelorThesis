import cv2
import numpy as np
import time


class Recorder(object):
    def __init__(self, camera=0, counter=0, width=800, height=600):
        # img size
        self.width = width
        self.height = height

        # capture camera input
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(3, width)
        self.cap.set(4, height)

        print(self.cap.get(3), self.cap.get(4))

        # capture size

        # counter and file/path settings
        self.counter = counter


    def capture(self, net, reaction_func):
        """
        continuously captures webcam feed

        :param reaction_func: a function that should react to some specific frame (on key press)
        :param net: the network that should process images 
        """
        while True:
            # capture each frame
            ret, frame = self.cap.read()

            # resize img, if cam does not support recording in a specific size
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))

            # frame operations
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # display frames
            cv2.imshow("frame (original)", frame)

            # listen to user input
            k = cv2.waitKey(1)  # 0 will wait for user input before recording a new frame

            # react to user input
            if k == ord('q'):
                break
            elif k == ord('s'):  # record frame on pressing "s"
                reaction_func(net, np.array([frame]))
                self.__increase_counter()


    def __increase_counter(self):
        self.counter += 1


    def __end_program(self):
        # When everything is done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

