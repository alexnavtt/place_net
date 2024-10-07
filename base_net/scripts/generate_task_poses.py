#!/usr/bin/python
import os
import math
import yaml
import torch
import random
import open3d
import argparse
import numpy as np
import scipy.spatial

from curobo.types.math import Pose as cuRoboPose
from curobo.types.robot import RobotConfig

from base_net.utils import task_visualization
from base_net.utils.base_net_config import BaseNetConfig

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

def are_spheres_in_collision(
        kd_tree: open3d.geometry.KDTreeFlann,
        ee_spheres: np.ndarray, 
        ee_position_in_world: np.ndarray, 
        ee_orientation_in_world: scipy.spatial.transform.Rotation, 
        inflation: float = 0.0
    ) -> bool:
    world_tform_ee = np.eye(4)
    world_tform_ee[:3, :3] = ee_orientation_in_world.as_matrix()
    world_tform_ee[:3,  3] = ee_position_in_world
    
    for sphere in ee_spheres:
        x, y, z, radius = sphere
        radius += inflation
        location_in_ee_frame = np.array([x, y, z, 1])
        location_in_world_frame = (world_tform_ee @ location_in_ee_frame)[:3]
        num_points_in_collision = kd_tree.search_radius_vector_3d(location_in_world_frame, radius)[0]
        if num_points_in_collision > 0:
            return True
        
    return False

def visualize_task_poses(pointcloud: open3d.geometry.PointCloud, surface_poses: cuRoboPose, close_poses: cuRoboPose, far_poses: cuRoboPose, robot_config: RobotConfig) -> None:
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. All input must be defined in the world frame
    """

    geometries = [{'geometry': pointcloud, 'name': 'environment'}]

    # Get the end effector collision spheres to make sure samples poses are valid
    ee_spheres = get_end_effector_spheres(robot_config).cpu().numpy()

    geometries = geometries + task_visualization.get_task_arrows(surface_poses, suffix='_surface')
    geometries = geometries + task_visualization.get_task_arrows(close_poses, suffix='_close')
    geometries = geometries + task_visualization.get_task_arrows(far_poses, suffix='_far')
    geometries = geometries + task_visualization.get_spheres(ee_spheres, surface_poses, color=[1.0, 0.5, 0.0])
    geometries = geometries + task_visualization.get_spheres(ee_spheres, close_poses, color=[1.0, 0.0, 0.5])
    geometries = geometries + task_visualization.get_spheres(ee_spheres, far_poses, color=[0.0, 1.0, 0.5])

    open3d.visualization.draw(geometry=geometries)

def sample_distant_poses(pointcloud: open3d.geometry.PointCloud, model_config: BaseNetConfig, count: int, min_offset: float, max_offset: float) -> cuRoboPose:
    """
    Generate poses at random locations in the environment and only accept those that contain obstacles 
    in a given 'maximum distance' sphere while having no obstacles within a 'minimum distance' sphere
    """
    position_tensor = torch.empty([0, 3])
    quaternion_tensor = torch.empty([0, 4])

    points_reminaing = count
    max_attempt_count = 10 * points_reminaing
    attempt_count = 0

    bounding_box = pointcloud.get_axis_aligned_bounding_box()
    bounding_box.max_bound = (bounding_box.max_bound[0], bounding_box.max_bound[1], model_config.model.workspace_height)
    bounding_box.min_bound = (bounding_box.min_bound[0], bounding_box.min_bound[1], model_config.model.workspace_floor)

    ee_spheres = get_end_effector_spheres(model_config.robot).cpu().numpy()
    kd_tree = open3d.geometry.KDTreeFlann(geometry=pointcloud)

    while points_reminaing > 0 and attempt_count < max_attempt_count:
        attempt_count += 1

        # Randomly sample a point within the pointcloud bounds
        fractions = np.random.rand(3)
        point = np.asarray(bounding_box.min_bound) + fractions * (np.asarray(bounding_box.max_bound) - np.asarray(bounding_box.min_bound)) 

        # Randomly sample an orientation with the Shoemaker's Algorithm (https://en.wikipedia.org/wiki/3D_rotation_group#Uniform_random_sampling)
        u1, u2, u3 = np.random.rand(3)

        c1 = math.sqrt(1 - u1)
        c2 = math.sqrt(u1)
        sin2 = math.sin(2*math.pi*u2)
        cos2 = math.cos(2*math.pi*u2)
        sin3 = math.sin(2*math.pi*u3)
        cos3 = math.cos(2*math.pi*u3)
        quat = np.array([c1*sin2, c1*cos2, c2*sin3, c2*cos3])

        # Check to make syre there are no points within the minimum bound
        if are_spheres_in_collision(
            kd_tree=kd_tree,
            ee_spheres=ee_spheres,
            ee_position_in_world=point,
            ee_orientation_in_world=scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True),
            inflation=min_offset
        ): continue 

        # Check to make sure there are points within the maximum bound
        if not are_spheres_in_collision(
            kd_tree=kd_tree,
            ee_spheres=ee_spheres,
            ee_position_in_world=point,
            ee_orientation_in_world=scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True),
            inflation=max_offset
        ): continue 

        # Append the pose to the samples
        position_tensor = torch.concatenate([position_tensor, torch.Tensor(point).unsqueeze(0)], dim=0)
        quaternion_tensor = torch.concatenate([quaternion_tensor, torch.Tensor(quat).unsqueeze(0)], dim=0)
        points_reminaing -= 1

    if points_reminaing != 0:
        print(f"WARNING: Out of the {model_config.task_generation.counts.close} points requested to be sampled, we could only find {model_config.task_generation.counts.close - points_reminaing}")

    return cuRoboPose(position_tensor.to(model_config.model.device), quaternion_tensor.to(model_config.model.device))

def sample_surface_poses(pointcloud: open3d.geometry.PointCloud, model_config: BaseNetConfig) -> cuRoboPose:
    """
    Generate poses which are very close to surfaces in the environment with the end effect
    oriented such that the x-axis is parallel to the surface normal. Roll is randomly assigned
    """
    position_tensor = torch.empty([0, 3])
    quaternion_tensor = torch.empty([0, 4])

    # Get the end effector collision spheres to make sure samples poses are valid
    ee_spheres = get_end_effector_spheres(model_config.robot).cpu().numpy()
    print(ee_spheres)

    # Get a KD tree for sphere-pointcloud collision detection
    kd_tree = open3d.geometry.KDTreeFlann(geometry=pointcloud)

    # List all points as available to sample
    available_point_labels = np.ones(len(pointcloud.points), dtype=bool)
    points_remaining = model_config.task_generation.counts.surface
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
        ee_location = point + normal * model_config.task_generation.offsets.surface_offset

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

        # Check for collisions with the environment
        if are_spheres_in_collision(
            kd_tree=kd_tree,
            ee_spheres=ee_spheres,
            ee_position_in_world=ee_location,
            ee_orientation_in_world=scipy.spatial.transform.Rotation.from_matrix(rot_mat),
            inflation=model_config.task_generation.offsets.surface_min
        ): continue

        # If not in collision, add this pose to the return set and decrement the remaining pose counter
        quaternion = scipy.spatial.transform.Rotation.from_matrix(rot_mat).as_quat(scalar_first=True)
        position_tensor = torch.concatenate([position_tensor, torch.Tensor(ee_location).unsqueeze(0)], dim=0)
        quaternion_tensor = torch.concatenate([quaternion_tensor, torch.Tensor(quaternion).unsqueeze(0)], dim=0)
        points_remaining -= 1

    return cuRoboPose(position_tensor.cuda(), quaternion_tensor.cuda())

def main():
    args = load_arguments()
    model_config = BaseNetConfig.from_yaml(args.config_file, load_tasks=False)
    task_config = model_config.task_generation

    for pointcloud_name, pointcloud in model_config.pointclouds.items():
        surface_poses = sample_surface_poses(pointcloud, model_config)
        close_poses = sample_distant_poses(pointcloud, model_config, task_config.counts.close, task_config.offsets.close_min, task_config.offsets.close_max)
        far_poses = sample_distant_poses(pointcloud, model_config, task_config.counts.far, task_config.offsets.far_min, task_config.offsets.far_max)

        sample_poses = cuRoboPose(
            position=torch.concatenate([surface_poses.position, close_poses.position, far_poses.position], dim=0),
            quaternion=torch.concatenate([surface_poses.quaternion, close_poses.quaternion, far_poses.quaternion], dim=0)
        )

        if args.output_path is not None:
            with open(os.path.join(args.output_path, f'{pointcloud_name}.task'), 'w') as f:
                tasks = []
                for position, orientation in zip(sample_poses.position, sample_poses.quaternion):
                    tasks.append({'position': position.cpu().numpy().tolist(), 'orientation': orientation.cpu().numpy().tolist()})
                yaml.dump(tasks, f)
        else:
            print("No output path provided, generated poses have not been saved")
    
        visualize_task_poses(pointcloud, surface_poses, close_poses, far_poses, model_config.robot)

if __name__ == "__main__":
    main()