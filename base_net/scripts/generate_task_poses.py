#!/usr/bin/python
import os
import yaml
import torch
import random
import open3d
import argparse
import numpy as np
import scipy.spatial

from urdf_parser_py.urdf import Robot, Link, Joint, Pose as urdfPose
from curobo.types.math import Pose as cuRoboPose
from curobo.types.robot import RobotConfig

from calculate_ground_truth import load_robot_config, load_pointclouds
import task_visualization

def load_arguments() -> dict:
    parser = argparse.ArgumentParser(
        prog="generate_task_poses.py",
        description="Script to sample task poses for a given set of pointcloud environments",
    )
    parser.add_argument('--config-file', default='../config/task_definitions.yaml', help='Config yaml file in which to look for a list of pointclouds')
    parser.add_argument('--output-path', help='Path to a folder in which to place the resulting task definitions, encoded as a YAML file')
    return parser.parse_args()

def get_end_effector_spheres(robot_config: RobotConfig) -> torch.Tensor:
    """
    Get all of the collision spheres on the end effector link and all links
    with fixed joints to the end effector link. The output tensor is in 
    the form [num_spheres, 4] with each sphere described as [x, y, z, r]
    """

    ee_link = robot_config.kinematics.kinematics_config.base_link
    transform_from_ee: dict[str, np.ndarray] = task_visualization.get_links_attached_to(ee_link, robot_config)
    
    ee_spheres = torch.empty([0, 4]).cuda(robot_config.kinematics.tensor_args.device)
    for link_name, link_transform in transform_from_ee.items():
        if link_name not in robot_config.kinematics.kinematics_config.link_name_to_idx_map:
            continue
        new_spheres = robot_config.kinematics.kinematics_config.get_link_spheres(link_name)
        num_spheres = new_spheres.size()[0]
        sphere_locs = torch.concatenate([new_spheres[:, :3], torch.ones(num_spheres, 1).to(new_spheres.device)], dim=1)
        transform_tensor = torch.Tensor(link_transform.T).to(new_spheres.device)
        sphere_locs = torch.matmul(sphere_locs, transform_tensor)
        new_spheres[:, :3] = sphere_locs[:, :3]
        ee_spheres = torch.concatenate([ee_spheres, new_spheres], dim=0)

    return ee_spheres

def visualize_task_poses(pointcloud: open3d.geometry.PointCloud, task_poses: cuRoboPose, robot_config: RobotConfig) -> None:
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. All input must be defined in the world frame
    """

    geometries = [pointcloud]

    # Get the end effector collision spheres to make sure samples poses are valid
    ee_spheres = get_end_effector_spheres(robot_config).cpu().numpy()

    geometries = geometries + task_visualization.get_task_arrows(task_poses)
    geometries = geometries + task_visualization.get_spheres(ee_spheres, task_poses)

    open3d.visualization.draw(geometry=geometries)

def sample_distant_poses(pointcloud: open3d.geometry.PointCloud, num_poses: int) -> cuRoboPose:
    pass

def sample_close_poses(pointcloud: open3d.geometry.PointCloud, num_poses: int) -> cuRoboPose:
    pass

def sample_surface_poses(pointcloud: open3d.geometry.PointCloud, num_poses: int, robot_config: RobotConfig, config: dict) -> cuRoboPose:
    """
    Generate poses which are very close to surfaces in the environment with the end effect
    oriented such that the x-axis is parallel to the surface normal. Roll is randomly assigned
    """
    position_tensor = torch.empty([0, 3])
    quaternion_tensor = torch.empty([0, 4])

    # Get the end effector collision spheres to make sure samples poses are valid
    ee_spheres = get_end_effector_spheres(robot_config).cpu().numpy()
    print(ee_spheres)

    # Get a KD tree for sphere-pointcloud collision detection
    kd_tree = open3d.geometry.KDTreeFlann(geometry=pointcloud)

    # List all points as available to sample
    available_point_labels = np.ones(len(pointcloud.points), dtype=bool)
    points_remaining = num_poses
    while points_remaining > 0 and any(available_point_labels):
        available_point_indices = np.nonzero(available_point_labels)[0]
        
        # Sample a point
        label_idx = random.randrange(len(available_point_indices))
        point_idx = available_point_indices[label_idx]
        available_point_labels[point_idx] = False

        # Retrieve the geometry
        point  = np.asarray(pointcloud.points)[point_idx, :]
        normal = np.asarray(pointcloud.normals)[point_idx, :]
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal) # just in case

        # Project the surface normal to the end effector location
        offset: float = config['close_point_offset']
        ee_location = point + normal * offset

        # Sample a random point in 3D space not along the normal
        while True:
            random_point = np.random.rand(1, 3)
            offset_vector = random_point - ee_location
            offset_vector = offset_vector/np.linalg.norm(offset_vector)
            if np.linalg.norm(np.cross(offset_vector, normal)) > 1e-3:
                break

        # Project this point onto the plane described by the point and the normal to define the poes y-axis
        pose_x_axis = -normal
        plane_vector = offset_vector - np.dot(offset_vector, pose_x_axis) * pose_x_axis
        pose_y_axis = plane_vector / np.linalg.norm(plane_vector)

        # Calculate the cross product to get the z-axis and create the rotation matrix
        pose_z_axis = np.cross(pose_x_axis, pose_y_axis)
        rot_mat = np.vstack([pose_x_axis, pose_y_axis, pose_z_axis]).T
        quaternion = scipy.spatial.transform.Rotation.from_matrix(rot_mat).as_quat(scalar_first=True)

        # Convert to a 4x4 homogenous transformation matrix for transform calculations
        world_tform_ee = np.eye(4)
        world_tform_ee[:3, :3] = rot_mat
        world_tform_ee[:3,  3] = ee_location.T

        # Check for collision with the end effector sphere
        in_collision = False
        for sphere in ee_spheres:
            x, y, z, radius = sphere
            location_in_ee_frame = np.array([x, y, z, 1])
            location_in_world_frame = (world_tform_ee @ location_in_ee_frame)[:3]
            num_points_in_collision = kd_tree.search_radius_vector_3d(location_in_world_frame, radius)[0]
            if num_points_in_collision > 0:
                in_collision = True
                break

        # If not in collision, add this pose to the return set and decrement the remaining pose counter
        if not in_collision:
            position_tensor = torch.concatenate([position_tensor, torch.Tensor(ee_location).unsqueeze(0)], dim=0)
            quaternion_tensor = torch.concatenate([quaternion_tensor, torch.Tensor(quaternion).unsqueeze(0)], dim=0)
            points_remaining -= 1

    return cuRoboPose(position_tensor.cuda(), quaternion_tensor.cuda())

def sample_poses_for_pointcloud(pointcloud: open3d.geometry.PointCloud, num_poses: int, split: dict) -> cuRoboPose:
    pass

def main():
    args = load_arguments()

    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error loading config file: {e}")
        
    # Load the pointclouds
    pointclouds = load_pointclouds(config)

    # Load the robot model
    robot_config = load_robot_config(config)

    pointcloud_name = 'URSA_sample_valves'
    test_pointcloud = pointclouds['URSA_sample_valves']
    sample_poses = sample_surface_poses(test_pointcloud, 300, robot_config, config)

    with open(os.path.join(args.output_path, f'{pointcloud_name}.task'), 'w') as f:
        tasks = []
        for position, orientation in zip(sample_poses.position, sample_poses.quaternion):
            tasks.append({'position': position.cpu().numpy().tolist(), 'orientation': orientation.cpu().numpy().tolist()})
        yaml.dump(tasks, f)
    
    visualize_task_poses(test_pointcloud, sample_poses, robot_config)

if __name__ == "__main__":
    main()