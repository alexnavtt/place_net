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

from tqdm import tqdm
from torch import Tensor
from curobo.types.robot import RobotConfig

from base_net.utils import task_visualization
from base_net.utils.base_net_config import BaseNetConfig
from base_net.utils.pointcloud_region import PointcloudRegion

def load_arguments() -> dict:
    parser = argparse.ArgumentParser(
        prog="generate_task_poses.py",
        description="Script to sample task poses for a given set of pointcloud environments",
    )
    parser.add_argument('--config-file', default='../config/task_definitions.yaml', help='Config yaml file in which to look for a list of pointclouds')
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

def visualize_task_poses(pointcloud: open3d.geometry.PointCloud, surface_poses: Tensor, close_poses: Tensor, far_poses: Tensor, robot_config: RobotConfig, regions: PointcloudRegion) -> None:
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
    geometries = geometries + task_visualization.get_spheres(ee_spheres, surface_poses, color=[1.0, 0.5, 0.0], label='surface_poses')
    geometries = geometries + task_visualization.get_spheres(ee_spheres, close_poses, color=[1.0, 0.0, 0.5], label='close_poses')
    geometries = geometries + task_visualization.get_spheres(ee_spheres, far_poses, color=[0.0, 1.0, 0.5], label='far_poses')
    geometries += task_visualization.get_regions(regions)

    open3d.visualization.draw(geometry=geometries)

def sample_distant_poses(name: str, regions: PointcloudRegion, model_config: BaseNetConfig, sample_config: dict) -> Tensor:
    """
    Generate poses at random locations in the environment and only accept those that contain obstacles 
    in a given 'maximum distance' sphere while having no obstacles within a 'minimum distance' sphere
    """
    position_tensor = torch.empty([0, 3])
    quaternion_tensor = torch.empty([0, 4])

    points_sampled = 0
    max_attempt_count = 100 * sample_config['count']
    attempt_count = 0

    pointcloud = regions.pointcloud
    bounding_box = pointcloud.get_axis_aligned_bounding_box()
    min_bound = np.array(bounding_box.min_bound)
    min_bound[0] -= model_config.task_geometry.max_radial_reach
    min_bound[1] -= model_config.task_geometry.max_radial_reach

    max_bound = np.array(bounding_box.max_bound)
    max_bound[0] += model_config.task_geometry.max_radial_reach
    max_bound[1] += model_config.task_geometry.max_radial_reach
    max_bound[2] = max(model_config.task_geometry.base_link_elevation + model_config.task_geometry.max_radial_reach, max_bound[2])
    bounding_box = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    ee_spheres = get_end_effector_spheres(model_config.robot).cpu().numpy()
    kd_tree = open3d.geometry.KDTreeFlann(geometry=regions._pointcloud)

    while points_sampled < sample_config['count'] and attempt_count < max_attempt_count:
        attempt_count += 1

        # Randomly sample a point within the pointcloud bounds
        fractions = np.random.rand(3)
        point = np.asarray(bounding_box.min_bound) + fractions * (np.asarray(bounding_box.max_bound) - np.asarray(bounding_box.min_bound)) 
        if not regions.contains(point):
            continue

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
            inflation=sample_config['offset_bounds'][0]
        ): continue 

        # Check to make sure there are points within the maximum bound
        if not are_spheres_in_collision(
            kd_tree=kd_tree,
            ee_spheres=ee_spheres,
            ee_position_in_world=point,
            ee_orientation_in_world=scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True),
            inflation=sample_config['offset_bounds'][1]
        ): continue 

        # Append the pose to the samples
        position_tensor = torch.concatenate([position_tensor, torch.tensor(point).unsqueeze(0)], dim=0)
        quaternion_tensor = torch.concatenate([quaternion_tensor, torch.tensor(quat).unsqueeze(0)], dim=0)
        points_sampled += 1

    if points_sampled != sample_config['count']:
        print(f"WARNING: Out of the {sample_config['count']} {sample_config['name']} points requested to be sampled, we could only find {points_sampled}")

    print(f'{name} [{sample_config["name"]}]: {points_sampled}/{sample_config["count"]}')
    return torch.concatenate([position_tensor, quaternion_tensor], dim=1)

def sample_surface_poses(name: str, regions: PointcloudRegion, model_config: BaseNetConfig) -> Tensor:
    """
    Generate poses which are very close to surfaces in the environment with the end effect
    oriented such that the x-axis is parallel to the surface normal. Roll is randomly assigned
    """
    position_tensor = torch.empty([0, 3])
    quaternion_tensor = torch.empty([0, 4])

    # Get the end effector collision spheres to make sure samples poses are valid
    ee_spheres = get_end_effector_spheres(model_config.robot).cpu().numpy()

    # Get a KD tree for sphere-pointcloud collision detection
    pointcloud = regions.pointcloud
    kd_tree = open3d.geometry.KDTreeFlann(geometry=regions._pointcloud)
    sampled_points = open3d.geometry.PointCloud()

    # Define a function for checking if a point is within bounds of the pointcloud itself
    # This helps to minimize poses sampled outside trimmed portions of the pointcloud
    pointcloud_bbox = open3d.geometry.OrientedBoundingBox.create_from_points(regions._pointcloud.points)
    def is_in_pointcloud_bbox(point) -> bool:
        point_vec = open3d.utility.Vector3dVector([point])
        return len(pointcloud_bbox.get_point_indices_within_bounding_box(point_vec)) > 0

    # List all points as available to sample
    available_point_labels = np.ones(len(pointcloud.points), dtype=bool)
    points_sampled = 0
    with tqdm(total=model_config.task_generation.surface_point_count) as pbar:
        while points_sampled < model_config.task_generation.surface_point_count and any(available_point_labels):
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
            ee_location = point + normal * model_config.task_generation.surface_point_offset
            if not regions.contains(ee_location) or not is_in_pointcloud_bbox(ee_location):
                continue

            # Check to see that there aren't too many sampled points already close to this one
            sampled_points.points.extend(open3d.utility.Vector3dVector([ee_location]))
            search = open3d.geometry.KDTreeFlann(geometry=sampled_points)
            # TODO: Make this not hard-coded
            num_close_samples = search.search_radius_vector_3d(ee_location, 0.2)[0]
            if num_close_samples > 3:
                continue

            # Sample a random point in 3D space not along the normal
            while True:
                offset_vector = np.random.rand(1, 3)
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
                inflation=model_config.task_generation.surface_point_clearance
            ): continue

            # If not in collision, add this pose to the return set and decrement the remaining pose counter
            quaternion = scipy.spatial.transform.Rotation.from_matrix(rot_mat).as_quat(scalar_first=True)
            position_tensor = torch.concatenate([position_tensor, torch.Tensor(ee_location).unsqueeze(0)], dim=0)
            quaternion_tensor = torch.concatenate([quaternion_tensor, torch.Tensor(quaternion).unsqueeze(0)], dim=0)
            points_sampled += 1
            pbar.update(1)

    print(f'{name} [Surface]: {points_sampled}/{model_config.task_generation.surface_point_count}')
    return torch.concatenate([position_tensor, quaternion_tensor], dim=1)

def main():
    args = load_arguments()
    model_config = BaseNetConfig.from_yaml_file(args.config_file, load_tasks=False)
    task_config = model_config.task_generation

    # If we are checking collisions then we need to sample based on the pointcloud geometry
    for pointcloud_name, original_pointcloud in model_config.pointclouds.items():
        regions = model_config.task_generation.regions[pointcloud_name]

        sampled_poses_list = [sample_surface_poses(pointcloud_name, regions, model_config)]
        for offset_sample_config in task_config.offset_points:
            sampled_poses_list += [sample_distant_poses(pointcloud_name, regions, model_config, offset_sample_config)]

        sample_poses: Tensor = torch.concatenate(sampled_poses_list, dim=0)

        if model_config.task_path is not None:
            filename = os.path.join(model_config.task_path, f'{pointcloud_name}_task.pt')
            torch.save(sample_poses, filename)
            print(f'Saved {sample_poses.size(0)} task poses to {filename}')
        else:
            print("No output path provided, generated poses have not been saved")
    
        if model_config.task_generation.visualize:
            visualize_task_poses(original_pointcloud, *sampled_poses_list, model_config.robot, regions)

if __name__ == "__main__":
    main()