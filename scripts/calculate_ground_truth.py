#!/usr/bin/python3

# std library
import os
import copy
import math
import time
import argparse
import open3d.visualization
from typing_extensions import TypeAlias

# 3rd party minor
import yaml
import scipy.spatial
import open3d
import numpy as np

# Allow running even without ROS
try:
    from ament_index_python import get_package_share_directory
except ModuleNotFoundError:
    pass

# pytorch
import torch
from torch import Tensor

# curobo
from curobo.types.math import Pose as cuRoboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig, CudaRobotModel
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
from curobo.geom.types import WorldConfig, Mesh

cuRoboTransform: TypeAlias = cuRoboPose

# base_net
from invert_robot_model import main as invert_urdf
DEVICE = torch.device('cuda', 0)

def load_config():
    """
    Load the path to the config file from runtime arguments and load the config as a dictionary
    """

    global DEVICE
    parser = argparse.ArgumentParser(
        prog="calculate_ground_truth.py",
        description="Script to calculate the ground truth reachability values for BaseNet",
    )
    parser.add_argument('--config-file', default='../config/task_definitions.yaml', help='configuration yaml file for the robot and task definitions')
    args = parser.parse_args()

    config = load_yaml(args.config_file)
    if 'cuda_device' in config.keys() and config['cuda_device'] is not None:
        DEVICE = torch.device('cuda', int(config['cuda_device']))

    return config

def load_pointclouds(config: dict) -> dict[str: open3d.geometry.PointCloud]:
    """
    Load the pointclouds to be used in calculations from the file paths specified in 
    the config file. One extra pointcloud with key 'empty' is also added with no points
    """

    pointclouds = {'empty': open3d.geometry.PointCloud()}
    if 'pointclouds' not in config:
        return pointclouds
    
    height_filter = open3d.geometry.AxisAlignedBoundingBox(
        min_bound=[-1e10, -1e10, 0.05], 
        max_bound=[1e10, 1e10, config['end_effector_elevation'] + config['robot_reach_radius'] + 0.2]
    )

    for item in config['pointclouds']:
        file_path, elevation = item['path'], item['elevation']

        _, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)

        if os.path.isfile(file_path) and extension == '.pcd':
            if ' ' in name:
                raise RuntimeError(f'File name "{name}" contains spaces, which is incompatible with this script')
            pointcloud_o3d = open3d.io.read_point_cloud(file_path)
            if not pointcloud_o3d.has_normals():
                pointcloud_o3d.estimate_normals()
                # raise RuntimeError("Cannot operate on pointclouds without normals")
            pointcloud_o3d.translate([0, 0, -elevation])
            pointclouds[name] = pointcloud_o3d.crop(height_filter)

    return pointclouds
                
def load_robot_config(config: dict) -> RobotConfig:
    """
    Load the cuRobo config from the config yaml/XRDF file and the urdf specified in the config.
    This function inverts the loaded URDF such that the end effector becomes the base link 
    and the base link becomes the end effector. No modifications are needed from the user
    to either the URDF or the cuRobot config file.
    """
    
    # Resolve ros package paths if necessary
    if 'ros_package' in config['curobo_config_file']:
        curobo_file = os.path.join(
            get_package_share_directory(config['curobo_config_file']['ros_package']),
            config['curobo_config_file']['path']
        )
    else:
        curobo_file = config['curobo_config_file']['path']

    urdf_config = config['urdf_file']
    if 'ros_package' in urdf_config:
        urdf_file = os.path.join(
            get_package_share_directory(urdf_config['ros_package']),
            urdf_config['path']
        )
    else:
        urdf_file = urdf_config['path']
    
    # Load and process the cuRobo config file
    curobo_config_extension = os.path.splitext(curobo_file)[1]
    robot_config = load_yaml(curobo_file)
    if curobo_config_extension == '.xrdf':
        ee_link = robot_config['tool_frames'][0]
        for config_item in robot_config['modifiers']:
            if 'set_base_frame' in config_item:
                base_link = config_item['set_base_frame']
                config_item['set_base_frame'] = ee_link
                robot_config['tool_frames'][0] = base_link
                break

        # Temporary workaround because cuRobo doesn't properly process an XRDF dict
        with open('/tmp/robot_xrdf.xrdf', 'w') as f:
            yaml.dump(robot_config, f)
        robot_config = '/tmp/robot_xrdf.xrdf'

    elif curobo_config_extension == '.yaml':
        ee_link = robot_config['robot_cfg']['kinematics']['ee_link']
        base_link = robot_config['robot_cfg']['kinematics']['base_link']
        robot_config['robot_cfg']['kinematics']['ee_link'] = base_link
        robot_config['robot_cfg']['kinematics']['base_link'] = ee_link
    else:
        raise RuntimeError(f'Received cuRobo config file with unsupported extension: "{curobo_config_extension}"')

    # Load and process the URDF file
    invert_urdf(urdf_file, urdf_config['xacro_args'] if 'xacro_args' in urdf_config else '', ee_link, "/tmp/inverted_urdf.urdf")

    return RobotConfig(kinematics=CudaRobotModelConfig.from_robot_yaml_file(
        file_path=robot_config,
        ee_link=base_link,
        urdf_path="/tmp/inverted_urdf.urdf",
        tensor_args=TensorDeviceType(device=DEVICE))
    )

def load_ik_solver(robot_config: RobotConfig, pointcloud: Tensor, crop_radius: float):
    """
    Consolidate the robot config and environment data to create a collision-aware IK
    solver for this particular environment. For efficiency, the pointcloud is cropped
    to only those points which are reachable from the task pose by the robot collision 
    bodies
    """
    start = time.perf_counter()
    tensor_args = TensorDeviceType(device=DEVICE)

    bound = np.array([crop_radius]*3)
    crop_box = open3d.geometry.AxisAlignedBoundingBox(min_bound=-bound, max_bound=bound)
    pointcloud = pointcloud.crop(crop_box)

    open3d_mesh = open3d.geometry.TriangleMesh()
    if len(pointcloud.points) > 0:
        world_mesh = Mesh.from_pointcloud(pointcloud=np.asarray(pointcloud.points), pitch=0.05)
        world_config = WorldConfig(
            mesh=[world_mesh]
        )
        trimesh = world_mesh.get_trimesh_mesh()
        open3d_mesh.vertices.extend(trimesh.vertices)
        open3d_mesh.triangles.extend(trimesh.faces)
        open3d_mesh.vertex_colors.extend(np.random.rand(len(trimesh.vertices), 3))
    else:
        world_config = None

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_config,
        world_config,
        rotation_threshold=0.1,
        position_threshold=0.01,
        num_seeds=10,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    ) 

    end = time.perf_counter()
    print(f"Loaded solver in {end-start} seconds")
    return IKSolver(ik_config), open3d_mesh

def load_base_pose_array(extent: float, num_pos: int = 20, num_yaws: int = 20) -> cuRoboPose:
    """
    Define the array of manipulator base-link poses for which we are trying to solve the
    reachability problem. The resulting array is defined centered around the origin and 
    aligned with the gravity-aligned task frame
    """
    yaw_angles = torch.arange(0, 2*math.pi - 1e-4, 2*math.pi/num_yaws)
    quats = torch.zeros([num_yaws, 4])
    quats[:, 0] = torch.cos(yaw_angles/2)
    quats[:, 3] = torch.sin(yaw_angles/2)

    cell_size = extent/num_pos
    x_coords = torch.arange(-extent/2, extent/2 - 1e-4, cell_size) + cell_size/2
    y_coords = torch.arange(-extent/2, extent/2 - 1e-4, cell_size) + cell_size/2

    x_coords_arranged = x_coords.repeat(num_pos)
    y_coords_arranged = y_coords.repeat_interleave(num_pos)

    pos_grid = torch.concatenate([x_coords_arranged.unsqueeze(1), y_coords_arranged.unsqueeze(1), torch.zeros([num_pos**2, 1])], dim=1)

    pos_grid_arranged = pos_grid.repeat_interleave(num_yaws, dim=0)
    yaw_grid_arranged = quats.repeat([num_pos**2, 1])

    curobo_pose = cuRoboPose(position=pos_grid_arranged.cuda(DEVICE), quaternion=yaw_grid_arranged.cuda(DEVICE))

    return curobo_pose

def flatten_task(task: cuRoboPose):
    """
    Given a pose in 3D space, return a pose at the same position with roll and 
    pitch components of the orientation removed
    """
    qw, qx, qy, qz = task.quaternion.squeeze().cpu().numpy()

    # Yaw calculation
    qwz = qw*qz
    qxy = qx*qy
    qyy = qy*qy
    qzz = qz*qz

    sin_yaw_cos_pitch = 2*(qwz + qxy)
    cos_yaw_cos_pitch = 1 - 2*(qyy + qzz)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    # Reconstruct the quaternion with only the yaw component
    flattened_quaternion = np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])

    flattened_task = copy.deepcopy(task)
    flattened_task.quaternion[0,:] = Tensor(flattened_quaternion) 
    return flattened_task

def visualize_task(task_pose: cuRoboPose, pointcloud: open3d.geometry.PointCloud, base_poses: cuRoboPose):
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. All input must be defined in the world frame
    """

    task_arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.03,
        cone_radius=0.05,
        cylinder_height=0.2,
        cone_height=0.1,
    )

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
    rotation = scipy.spatial.transform.Rotation.from_quat(quat=task_pose.quaternion.squeeze().cpu().numpy(), scalar_first=True)
    task_arrow.rotate(rotation_to_x, center=[0, 0, 0])
    task_arrow.rotate(rotation.as_matrix(), center=[0, 0, 0])
    task_arrow.translate(task_pose.position.squeeze().cpu().numpy())
    task_arrow.paint_uniform_color([1, 0, 0])

    geometries = [task_arrow, pointcloud]

    for position, rotation in zip(base_poses.position, base_poses.quaternion):
        new_arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.03,
            cone_height=0.015,
            cylinder_split=1,
            resolution=4
        )
        new_arrow.paint_uniform_color([0, 0, 1])
        new_arrow.rotate(rotation_to_x, center=[0, 0, 0])
        new_arrow.rotate(scipy.spatial.transform.Rotation.from_quat(rotation.cpu().numpy(), scalar_first=True).as_matrix(), center=[0, 0, 0])
        new_arrow.translate(position.cpu().numpy())
        geometries.append(new_arrow)

    open3d.visualization.draw(geometry=geometries)

def visualize_solution(world_mesh: open3d.geometry.TriangleMesh, solution: IKResult, goal_poses: cuRoboPose, robot_model: CudaRobotModel):
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. Reachable base link poses will be colored green, 
    and unreachable ones will be colored red. If one exists, a random valid robot configuration
    will also be rendered. All input must be defined in the task frame
    """
    geometries = [world_mesh] if len(world_mesh.vertices) > 0 else []

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
    for position, rotation, success in zip(goal_poses.position, goal_poses.quaternion, solution.success):
        new_arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.03,
            cone_height=0.015,
            cylinder_split=1,
            resolution=4
        )
        new_arrow.paint_uniform_color([float(1-success.item()), success.item(), 0])
        new_arrow.rotate(rotation_to_x, center=[0, 0, 0])
        new_arrow.rotate(scipy.spatial.transform.Rotation.from_quat(rotation.cpu().numpy(), scalar_first=True).as_matrix(), center=[0, 0, 0])
        new_arrow.translate(position.cpu().numpy())
        geometries.append(new_arrow)

    # Render one of the successful poses randomly
    if torch.sum(solution.success) > 0:
        solution_idx = int(np.random.rand() * torch.sum(solution.success))
        robot_spheres = robot_model.get_robot_as_spheres(q=solution.solution[solution.success, :][solution_idx, :])[0]
        robot_spheres_o3d = [open3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius) for sphere in robot_spheres]
        for robot_sphere_o3d, robot_sphere in zip(robot_spheres_o3d, robot_spheres):
            robot_sphere_o3d.translate(robot_sphere.position)
            robot_sphere_o3d.vertex_colors.extend(np.random.rand(len(robot_sphere_o3d.vertices), 1).repeat(3, axis=1))
            geometries.append(robot_sphere_o3d)

    task_arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.015,
        cone_radius=0.03,
        cylinder_height=0.15,
        cone_height=0.07
    )
    task_arrow.paint_uniform_color([0, 0, 1])
    task_arrow.rotate(rotation_to_x, center=[0, 0, 0])
    geometries.append(task_arrow)

    open3d.visualization.draw(geometry=geometries)

def main():
    config = load_config()
    pointclouds = load_pointclouds(config)
    robot_config = load_robot_config(config)

    robot_model = CudaRobotModel(config=robot_config.kinematics)

    base_poses_in_flattened_task_frame = load_base_pose_array(
        extent=2*config['robot_reach_radius'], 
        num_pos=config['position_count'], 
        num_yaws=config['yaw_count']
    )
    num_poses = base_poses_in_flattened_task_frame.batch

    for task in config['tasks']:
        task_pointcloud_name = task['pointcloud']
        task_pose_in_world = cuRoboPose(position=Tensor(task['position']).to(DEVICE), quaternion=Tensor(task['orientation']).to(DEVICE))
        task_pose_in_world.quaternion[0,:] = torch.rand([1, 4])
        task_pose_in_world.quaternion[0,:] /= torch.norm(task_pose_in_world.quaternion.squeeze())

        # Transform the base poses from the flattened task frame to the task frame
        flattened_task_pose = flatten_task(task_pose_in_world)

        # Assign transform names for clarity of calculations
        world_tform_flattened_task = flattened_task_pose
        world_tform_task = task_pose_in_world
        task_tform_world: cuRoboTransform = world_tform_task.inverse()

        base_poses_in_world = world_tform_flattened_task.repeat(num_poses).multiply(base_poses_in_flattened_task_frame)
        base_poses_in_world.position[:,2] = config['end_effector_elevation']
        # visualize_task(task_pose_in_world, pointcloud_in_world, base_poses_in_world)

        # Transform the pointcloud from the world frame to the task frame
        task_R_world = scipy.spatial.transform.Rotation.from_quat(quat=task_tform_world.quaternion.squeeze().cpu().numpy(), scalar_first=True).as_matrix()
        task_tform_world_mat = np.eye(4)
        task_tform_world_mat[:3, :3] = task_R_world
        task_tform_world_mat[:3, 3] = task_tform_world.position.squeeze().cpu().numpy()

        pointcloud_in_world = pointclouds[task_pointcloud_name]
        pointcloud_in_task = pointcloud_in_world.transform(task_tform_world_mat)

        base_poses_in_task = task_tform_world.repeat(num_poses).multiply(base_poses_in_world)

        ik_solver, world_mesh = load_ik_solver(robot_config, pointcloud_in_task, config['obstacle_inclusion_radius'])
        solutions = ik_solver.solve_batch(goal_pose=base_poses_in_task)
        print(f"There are {torch.sum(solutions.success)} successful poses")
        print(f'Solved {base_poses_in_world.position.size()[0]} IK problems in {solutions.solve_time} seconds')
        visualize_solution(world_mesh, solutions, base_poses_in_task, robot_model)

if __name__ == "__main__":
    main()