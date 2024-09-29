import open3d
import argparse
from curobo.types.math import Pose as cuRoboPose

def load_config() -> dict:
    parser = argparse.ArgumentParser(
        prog="generate_task_poses.py",
        description="Script to sample task poses for a given set of pointcloud environments",
    )
    parser.add_argument('--config-file', default='../config/task_definitions.yaml', help='Config yaml file in which to look for a list of pointclouds')
    parser.add_argument('--output-path', help='Path to a folder in which to place the resulting task definitions, encoded as a YAML file')
    return parser.parse_args()

def sample_distant_poses(pointcloud: open3d.geometry.PointCloud, num_poses: int) -> cuRoboPose:
    pass

def sample_close_poses(pointcloud: open3d.geometry.PointCloud, num_poses: int) -> cuRoboPose:
    pass

def sample_surface_poses(pointcloud: open3d.geometry.PointCloud, num_poses: int) -> cuRoboPose:
    pass

def sample_poses_for_pointcloud(pointcloud: open3d.geometry.PointCloud, num_poses: int, split: dict) -> cuRoboPose:
    pass

def main():
    pass

if __name__ == "__main__":
    main()