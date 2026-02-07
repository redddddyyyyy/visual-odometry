import cv2
import numpy as np
import os
from src import VisualOdometry, TrajectoryVisualizer


def load_kitti_calibration(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('P0:'):
            data = line.strip().split()[1:]
            P = np.array([float(x) for x in data]).reshape(3, 4)
            K = P[:3, :3]
            return K
    return None


def main():
    data_dir = "data/kitti/00"

    calib_path = os.path.join(data_dir, "calib.txt")
    image_dir = os.path.join(data_dir, "image_0")

    K = load_kitti_calibration(calib_path)
    if K is None:
        print("Failed to load calibration")
        return

    print(f"Camera matrix K:\n{K}")

    vo = VisualOdometry(K)

    image_files = sorted(os.listdir(image_dir))
    num_frames = min(500, len(image_files))

    print(f"Processing {num_frames} frames...")

    for i in range(num_frames):
        img_path = os.path.join(image_dir, image_files[i])
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            continue

        pose = vo.process_frame(frame)

        if pose is not None and i % 50 == 0:
            x, z = pose[0, 3], pose[2, 3]
            print(f"Frame {i}: position = ({x:.2f}, {z:.2f})")

    trajectory = vo.get_trajectory()
    print(f"Total poses: {len(trajectory)}")

    viz = TrajectoryVisualizer()
    viz.plot_trajectory(trajectory)
    viz.save("assets/trajectory.png")
    viz.show()


if __name__ == "__main__":
    main()
