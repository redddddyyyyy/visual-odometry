import cv2
import numpy as np
import os
from src import VisualOdometry, TrajectoryVisualizer


def create_test_frames(num_frames=100):
    """Create synthetic frames with moving features for testing."""
    frames = []
    np.random.seed(42)

    # Create random feature positions
    num_points = 200
    base_points = np.random.randint(50, 550, (num_points, 2))

    for i in range(num_frames):
        frame = np.zeros((600, 800), dtype=np.uint8)

        # Simulate forward motion (points spread outward)
        scale = 1 + i * 0.005
        shift_x = i * 2

        for px, py in base_points:
            # Add some motion
            new_x = int((px - 400) * scale + 400 + np.random.randint(-2, 3))
            new_y = int((py - 300) * scale + 300 + np.random.randint(-2, 3))

            if 10 < new_x < 790 and 10 < new_y < 590:
                cv2.circle(frame, (new_x, new_y), 3, 255, -1)

        # Add some texture
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        frames.append(frame)

    return frames


def main():
    print("Visual Odometry - Simple Test")
    print("=" * 40)

    # Default camera matrix (approximate)
    K = np.array([
        [718.856, 0, 400],
        [0, 718.856, 300],
        [0, 0, 1]
    ], dtype=np.float64)

    print(f"Camera matrix K:\n{K}\n")

    # Create test frames
    print("Creating synthetic test frames...")
    frames = create_test_frames(num_frames=100)
    print(f"Created {len(frames)} frames\n")

    # Run VO
    vo = VisualOdometry(K)

    print("Processing frames...")
    for i, frame in enumerate(frames):
        pose = vo.process_frame(frame)

        if pose is not None and i % 20 == 0:
            x, z = pose[0, 3], pose[2, 3]
            print(f"  Frame {i}: position = ({x:.2f}, {z:.2f})")

    # Get results
    trajectory = vo.get_trajectory()
    print(f"\nTotal poses: {len(trajectory)}")

    # Visualize
    print("\nGenerating trajectory plot...")
    viz = TrajectoryVisualizer()
    viz.plot_trajectory(trajectory, title="Visual Odometry - Synthetic Test")
    viz.save("assets/trajectory_test.png")
    print("Saved to assets/trajectory_test.png")
    viz.show()


if __name__ == "__main__":
    main()
