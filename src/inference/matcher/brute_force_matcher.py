import cv2 as cv
import numpy as np

class BruteForceMatcher(object):
    def __init__(self, **kwargs):
        self.matcher = cv.BFMatcher()

    def __call__(self, des1, des2, **kwargs):
        matches = self.matcher.knnMatch(des1, des2, k=kwargs.get("k", 2))
        good = []
        for m,n in matches:
            if m.distance < kwargs.get("ratio", 0.75) * n.distance:
                good.append([m.queryIdx, m.trainIdx])

        return np.array(good)
