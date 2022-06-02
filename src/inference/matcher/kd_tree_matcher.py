import cv2 as cv
import numpy as np

class KDTreeMatcher(object):
    index_params = {
        "algorithm": 1,
        "trees": 1,
    }
    search_params = {
        "checks": 50,
    }

    def __init__(self, **kwargs):
        self.index_params.update(kwargs)
        self.search_params.update(kwargs)
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def __call__(self, des1, des2, **kwargs):
        matches = self.matcher.knnMatch(des1, des2, k=kwargs.get("k", 2))
        good = []
        for m,n in matches:
            if m.distance < kwargs.get("ratio", 0.75) * n.distance:
                good.append([m.queryIdx, m.trainIdx])

        return np.array(good)
