import cv2 as cv
import numpy as np

class SiftDetector(object):
    def __init__(self):
        self.sift = cv.SIFT.create()

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        kp, des = self.sift.detectAndCompute(image, None)

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "keypoints": np.array([list(x.pt) for x in kp]),
            "descriptors": np.array(des),
        }
        return ret_dict