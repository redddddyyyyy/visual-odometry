import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self, num_features=2000):
        self.orb = cv2.ORB_create(nfeatures=num_features)

    def detect_and_compute(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
