from enum import Enum
from detector.sift_detector import SiftDetector
from detector.superpoint_detector import SuperPointDetector
from detector.orb_detector import OrbDDetector
import numpy as np

class FeatureDetector(object):
    class FeatureType(Enum):
        SUPERPOINT = "superpoint"
        SIFT = "sift"
        ORB = "orb"
        def __str__(self) -> str:
            return self.value

    type2detector = {
        FeatureType.SUPERPOINT: SuperPointDetector,
        FeatureType.SIFT: SiftDetector,
        FeatureType.ORB: OrbDDetector,
    }
    def __init__(self, type: FeatureType):
        self.detector = self.type2detector[type]()
        # self.detector(np.zeros((1080, 1920, 3))) # warmup

    def __call__(self, image):
        return self.detector(image)