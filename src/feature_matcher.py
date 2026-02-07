import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75):
        self.ratio_threshold = ratio_threshold
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        return good_matches
