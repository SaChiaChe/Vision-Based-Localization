from enum import Enum
from matcher.brute_force_matcher import BruteForceMatcher
from matcher.kd_tree_matcher import KDTreeMatcher

class FeatureMatcher(object):
    class MatchMethod(Enum):
        BRUTE_FORCE = "brute_force"
        KD_TREE = "kd_tree"
        def __str__(self) -> str:
            return self.value

    method2matcher = {
        MatchMethod.BRUTE_FORCE: BruteForceMatcher,
        MatchMethod.KD_TREE: KDTreeMatcher,
    }

    def __init__(self, method: MatchMethod, **kwargs):
        self.matcher = self.method2matcher[method](**kwargs)

    def __call__(self, des1, des2, **kwargs):
        matches = self.matcher(des1, des2, **kwargs)
        return matches