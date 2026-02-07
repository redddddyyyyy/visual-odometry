import cv2
import numpy as np


class MotionEstimator:
    def __init__(self, K):
        self.K = K

    def estimate(self, kp1, kp2, matches):
        if len(matches) < 5:
            return None, None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        return R, t, mask
