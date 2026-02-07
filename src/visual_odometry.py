import numpy as np
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .motion_estimator import MotionEstimator


class VisualOdometry:
    def __init__(self, K):
        self.K = K
        self.extractor = FeatureExtractor(num_features=2000)
        self.matcher = FeatureMatcher(ratio_threshold=0.75)
        self.motion_estimator = MotionEstimator(K)

        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]

        self.prev_keypoints = None
        self.prev_descriptors = None

    def process_frame(self, frame):
        keypoints, descriptors = self.extractor.detect_and_compute(frame)

        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None

        matches = self.matcher.match(self.prev_descriptors, descriptors)
        R, t, mask = self.motion_estimator.estimate(self.prev_keypoints, keypoints, matches)

        if R is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.current_pose

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        self.current_pose = self.current_pose @ T
        self.trajectory.append(self.current_pose.copy())

        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return self.current_pose

    def get_trajectory(self):
        return self.trajectory
