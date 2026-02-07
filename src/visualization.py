import numpy as np
import matplotlib.pyplot as plt


class TrajectoryVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def plot_trajectory(self, trajectory, ground_truth=None, title="Visual Odometry Trajectory"):
        x_est = [T[0, 3] for T in trajectory]
        z_est = [T[2, 3] for T in trajectory]

        self.ax.plot(x_est, z_est, 'b-', label='Estimated', linewidth=2)
        self.ax.plot(x_est[0], z_est[0], 'go', markersize=10, label='Start')
        self.ax.plot(x_est[-1], z_est[-1], 'ro', markersize=10, label='End')

        if ground_truth is not None:
            x_gt = [pose[0, 3] for pose in ground_truth]
            z_gt = [pose[2, 3] for pose in ground_truth]
            self.ax.plot(x_gt, z_gt, 'g--', label='Ground Truth', linewidth=2)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Z (meters)')
        self.ax.set_title(title)
        self.ax.legend()
        self.ax.grid(True)
        self.ax.axis('equal')

    def show(self):
        plt.show()

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {path}")
