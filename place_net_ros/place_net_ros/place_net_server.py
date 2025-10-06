import os
import time
import math
import torch
import open3d
import numpy as np
from torch import Tensor
from threading import Thread

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseArray
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, read_points, structured_to_unstructured

from curobo.types.math import Pose as cuRoboPose
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolverConfig, IKSolver
from curobo.geom.types import WorldConfig, Mesh
from place_net.models.place_net import PlaceNet
from place_net.utils.place_net_config import PlaceNetConfig
from place_net_msgs.srv import QueryBaseLocation, QueryReachablePoses
from place_net.utils import geometry, pose_scorer, inverse_reachability_map
from place_net.scripts.calculate_ground_truth import solve_batched_ik, get_ground_truth_tensor

from .place_net_visualizer import PlaceNetVisualizer
from . import place_net_conversions
from .place_net_ros_parameters import place_net_ros_params

class PoseGrid:
    def __init__(self, x_range: float, y_range: float, x_res: int, y_res: int, yaw_res: int, z_elevation: float, device):
        self.x_range = x_range
        self.y_range = y_range
        self.x_res = x_res
        self.y_res = y_res
        self.yaw_res = yaw_res
        self.device = device

        self.poses = geometry.load_base_pose_array(x_range/2, y_range/2, x_res, y_res, yaw_res, device=device)
        self.poses.position[:, 2] = z_elevation
        min_grid_x, min_grid_y = torch.amin(self.poses.position[:, :2], dim=0)
        max_grid_x, max_grid_y = torch.amax(self.poses.position[:, :2], dim=0)
        
        self.lower_bound = torch.tensor([min_grid_x, min_grid_y], device=device)
        self.upper_bound = torch.tensor([max_grid_x, max_grid_y], device=device)
        self.extent = torch.tensor([max_grid_x-min_grid_x, max_grid_y-min_grid_y], device=device)
        self.grid_size = torch.tensor([x_res, y_res], device=device)

        self.scores = torch.zeros((y_res, x_res, yaw_res), dtype=torch.float, device=device)

    def translate(self, translation: Tensor) -> None:
        self.poses.position[:, :2] += translation
        self.lower_bound += translation
        self.upper_bound += translation

class PlaceNetServer(Node):
    def __init__(self):
        super().__init__(node_name='place_net_server')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Load the model from the checkpoint path
        param_listener = place_net_ros_params.ParamListener(self)
        self.params = param_listener.get_params()

        # Load the model if a checkpoint path is provided
        if self.params.checkpoint_path:
            base_path, _ = os.path.split(self.params.checkpoint_path)
            self.place_net_config = PlaceNetConfig.from_yaml_file(os.path.join(base_path, 'config.yaml'), load_pointclouds=False, load_solutions=False, load_tasks=False, device=self.params.device)
            if self.params.max_ik_count > 0:
                self.place_net_config.max_ik_count = self.params.max_ik_count
            self.place_net_model = PlaceNet(self.place_net_config)
            if self.params.compile:
                torch.set_float32_matmul_precision('high')
                self.place_net_model.compile()
                self.populate_master_score_grid = torch.compile(self.populate_master_score_grid)
                self.get_reachable_pose_indices = torch.compile(self.get_reachable_pose_indices)
        else:
            self.place_net_model = None

        # Load the inverse reachability map if a checkpoint path is provided
        if self.params.inverse_reachability_map_path:
            self.irm = inverse_reachability_map.InverseReachabilityMap.load(self.params.inverse_reachability_map_path)
            self.get_logger().info(f'Loaded inverse reachability solutions of shape {self.irm.solutions.size()} from {self.params.inverse_reachability_map_path}')
        else:
            self.irm = None

        self.pose_scorer = pose_scorer.PoseScorer(max_angular_window=torch.pi)

        checkpoint_config = torch.load(self.params.checkpoint_path, map_location=self.place_net_config.model.device, weights_only=True)
        if 'place_net_model' in checkpoint_config:
            self.place_net_model.load_state_dict(checkpoint_config['place_net_model'])
        else:
            self.place_net_model.load_state_dict(checkpoint_config['base_net_model'])
        self.place_net_model.eval()

        # Load the model geometry
        self.base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
            half_x_range=self.place_net_config.task_geometry.max_radial_reach,
            half_y_range=self.place_net_config.task_geometry.max_radial_reach,
            x_res=self.place_net_config.inverse_reachability.solution_resolution['x'],
            y_res=self.place_net_config.inverse_reachability.solution_resolution['y'],
            yaw_res=self.place_net_config.inverse_reachability.solution_resolution['yaw'],
            device=self.place_net_config.model.device
        )
        self.place_net_viz = PlaceNetVisualizer(self, self.place_net_config)

        # Start up the ROS service
        self.base_location_server = self.create_service(QueryBaseLocation, '~/query_base_location', self.base_location_callback)
        self.reachable_pose_server = self.create_service(QueryReachablePoses, '~/query_reachable_poses', self.reachable_poses_callback)
        self.memory_query_server = self.create_service(Trigger, '~/query_gpu_memory', self.gpu_memory_callback)

    def gpu_memory_callback(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        resp.message = torch.cuda.memory_summary(device=self.place_net_config.model.device)
        resp.success = True
        self.get_logger().info(resp.message)
        return resp

    def run_model(self, task_poses: Tensor, pointcloud: Tensor) -> Tensor:
        batch_size = task_poses.size(0)

        model_output = torch.zeros(
            batch_size, 
            self.place_net_config.inverse_reachability.solution_resolution['y'],
            self.place_net_config.inverse_reachability.solution_resolution['x'],
            self.place_net_config.inverse_reachability.solution_resolution['yaw'], 
            dtype=bool,
            device=self.place_net_config.model.device
        )
        
        with torch.no_grad():
            mini_batch_size = self.params.max_batch_size if self.params.max_batch_size > 0 else batch_size
            for index_start in range(0, batch_size, mini_batch_size):
                index_end = min(index_start + mini_batch_size, batch_size)
                size = index_end - index_start
                pointcloud_slice = [pointcloud]*size
                task_slice = task_poses[index_start:index_end]
                logits = self.place_net_model(pointcloud_slice, task_slice)
                model_output[index_start:index_end] = torch.sigmoid(logits) >= 0.5

        return model_output
    
    def get_solution_tensor(self, task_poses: Tensor, pointcloud: Tensor, mode: str) -> Tensor:
        """ Given a set of task poses, determine the binary map of reachable and not reachable base locations

        Args:
            task_poses [Tensor (n, 7)] : The set of task poses to try to reach defined in a gravity aligned frame
            pointcloud [Tensor (m, 3)] : A set of points to consider as obstacles. Ignore in IRM mode
            mode [str]: The method to use in determining the set of base locations.\nOptions are:
                        'model' - Use PlaceNet to determine base locations\n
                        'ground_truth' - Use cuRobo to perform a collision aware inverse reachability calculation\n
                        'irm' - Use a precomuted inverse reachability map. This method requires that the 
                                'inverse_reachability_map_path' ROS parameter be set with a valid IRM file
        Returns:
            [Tensor (n, nx, ny, ntheta)] Map of binary reachability values on a 3D SE2 pose grid
        """

        if mode == 'ground_truth':
            # We use the standard base pose array and master grid, instead of directly calculating the 
            # ground truth values for the master grid. This keeps results consistent between the model
            # and ground truth calculations
            base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
                half_x_range=self.place_net_config.task_geometry.max_radial_reach,
                half_y_range=self.place_net_config.task_geometry.max_radial_reach,
                x_res=self.place_net_config.inverse_reachability.solution_resolution['x'],
                y_res=self.place_net_config.inverse_reachability.solution_resolution['y'],
                yaw_res=self.place_net_config.inverse_reachability.solution_resolution['yaw'],
                device=self.place_net_config.model.device
            )

            # Convert the pointcloud to open3d
            pointcloud_z = pointcloud[:, 2]
            pointcloud = pointcloud[(pointcloud_z > self.place_net_config.task_geometry.min_pointcloud_elevation) & (pointcloud_z < self.place_net_config.task_geometry.max_pointcloud_elevation)]
            pointcloud_o3d = open3d.geometry.PointCloud()
            pointcloud_o3d.points.extend(pointcloud.cpu().numpy())
            return get_ground_truth_tensor(task_poses, pointcloud_o3d, base_poses_in_flattened_task_frame, self.place_net_config).to(self.place_net_config.model.device)
            
        elif mode == 'model':
            if self.place_net_model is None:
                raise RuntimeError('Received place_net model query but no model has been loaded!')
            result = self.run_model(task_poses, pointcloud)
            torch.cuda.empty_cache()
            return result
            
        elif mode == 'irm':
            if self.irm is None:
                raise RuntimeError('Received an inverse reachability map query but no IRM has been loaded!')
            if pointcloud.numel() > 0:
                self.get_logger().warn('A pointcloud was passed to an IRM base pose query. Collision avoidance for IRM queries is not supported')
            return self.irm.query_pose(task_poses.cpu()).to(self.place_net_config.model.device)

        else:
            raise RuntimeError(f'Unable to process base placement request for mode {mode}, options are ["model", "ground_truth", "irm"]')
    
    def create_master_score_grid(self, task_poses: Tensor) -> PoseGrid:
        min_x, min_y = torch.amin(task_poses[:, :2], dim=0)
        max_x, max_y = torch.amax(task_poses[:, :2], dim=0)

        x_cell_size: float = 2*self.place_net_config.task_geometry.max_radial_reach / (self.place_net_config.inverse_reachability.solution_resolution['x'] - 1)
        y_cell_size: float = 2*self.place_net_config.task_geometry.max_radial_reach / (self.place_net_config.inverse_reachability.solution_resolution['y'] - 1)

        x_range: float = max_x - min_x + 2*self.place_net_config.task_geometry.max_radial_reach
        y_range: float = max_y - min_y + 2*self.place_net_config.task_geometry.max_radial_reach
        x_res: int = math.floor(x_range / x_cell_size) + 1
        y_res: int = math.floor(y_range / y_cell_size) + 1
        yaw_res: int = self.place_net_config.inverse_reachability.solution_resolution['yaw']

        return PoseGrid(x_range, y_range, x_res, y_res, yaw_res, self.place_net_config.task_geometry.base_link_elevation, self.place_net_config.model.device)
    
    def populate_master_score_grid(self, master_grid: PoseGrid, task_poses: Tensor, pose_scores: Tensor) -> None:
        """ 
        Given the score grids for each task in its own frame, transfer that data to the master grid in the world frame
        
        Args:
            master_grid:    The object containing geometric and score information for the final grid of poses in the world frame. 
                            The scores will be populated after this function call
            task_poses:     The poses of the tasks in the world frame. Shape (B, 7)
            pose_scores:    The score associated with each (y, x, yaw) tuple in results for each task pose. Shape (B, ny, nx, ntheta)

        Returns:
            None
        """

        yaw_res: int = master_grid.yaw_res
        yaw_angles = geometry.extract_yaw_from_quaternions(task_poses[:, 3:])

        for task_pose, yaw_angle, layer_scores in zip(task_poses, yaw_angles, pose_scores):
            # Transform the results grid to this tasks base pose
            task_pose_curobo = cuRoboPose(position=task_pose[:3], quaternion=task_pose[3:])
            world_tform_flattened_task = geometry.flatten_task(task_pose_curobo)
            base_poses_in_world: cuRoboPose = world_tform_flattened_task.repeat(self.base_poses_in_flattened_task_frame.batch).multiply(self.base_poses_in_flattened_task_frame)

            # We only need to update entries that have reachable poses
            valid_model_indices = layer_scores.view(-1, yaw_res).sum(dim=1, dtype=bool)

            # Calculate the indices into the yaw angles
            yaw_index_offset: int = round(yaw_angle.item() / (2*math.pi / yaw_res))
            yaw_indices = torch.arange(yaw_res, device=self.place_net_config.model.device) + yaw_index_offset
            yaw_indices = torch.remainder(yaw_indices, yaw_res)
            yaw_indices = yaw_indices.long()

            # Calculate the indices into the positions
            xy_positions = base_poses_in_world.position[:, :2][::yaw_res]
            offsets = ((xy_positions - master_grid.lower_bound)) / master_grid.extent
            float_grid_indices = offsets * master_grid.grid_size
            grid_indices = torch.floor(float_grid_indices)

            # Calculate the offsets from the nearest grid cells
            fractional_offsets = torch.frac(float_grid_indices)
            for grid_offset in torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=self.place_net_config.model.device):
                # Get the indices for this particular corner of the grid cell
                offset_grid_indices = grid_indices + grid_offset
                valid_indices = ((offset_grid_indices >= 0) & (offset_grid_indices < master_grid.grid_size)).prod(dim=1, dtype=bool)
                valid_indices = valid_indices & valid_model_indices
                offset_grid_indices = offset_grid_indices[valid_indices]
                offset_grid_indices = offset_grid_indices[:, [1, 0]] # grid is arranged (y, x, yaw)
                offset_grid_indices = offset_grid_indices.long()

                # Weight using bilinear interpolation
                weight_components = torch.abs(fractional_offsets - grid_offset)
                weights = torch.prod(1 - weight_components, dim=-1)
                weights = weights[valid_indices]
                weights = weights.repeat_interleave(yaw_res)
                
                # Interleave the position and yaw indices
                yaw_indices_interleaved = yaw_indices.view(1, -1, 1).expand(offset_grid_indices.size(0), -1, 1)
                grid_indices_interleaved = offset_grid_indices.unsqueeze(1).expand(-1, yaw_res, 2)
                layer_indices = torch.concatenate([grid_indices_interleaved, yaw_indices_interleaved], dim=-1)
                layer_indices = layer_indices.view(-1, 3)

                # Assign to the score tensor
                y_indices = layer_indices[:, 0]
                x_indices = layer_indices[:, 1]
                t_indices = layer_indices[:, 2]
                
                valid_layer_scores = layer_scores.view(-1, yaw_res)[valid_indices, :].flatten()
                master_grid.scores[y_indices, x_indices, t_indices] += weights * valid_layer_scores

    def get_reachable_pose_indices(self, optimal_pose_in_world: cuRoboPose, task_poses_in_world: Tensor, reference_poses_in_task: cuRoboPose, valid_poses: Tensor) -> Tensor:
        """ Given the optimal pose, determine which task poses can be reached by the robot
        
        Args:
            - optimal_pose_in_world : The chosen base pose for the robot. Shape (1)
            - task_poses_in_world : The task poses for which we have calculated reachability. Shape (B, 7)
            - reference_poses_in_task : The array of sample poses reachability was calculated for, defined in the flattened task frame. Shape (ny*nx*ntheta)
            - valid_poses : The boolean model output indicating which poses can be reached. Shape (B, ny, nx, ntheta) 
        Returns:
            - A tensor of indices of the tasks that are reachable from the optimal pose
        """
        batch_size = task_poses_in_world.size(0)
        device = task_poses_in_world.device
        
        world_tform_task = cuRoboPose(position=task_poses_in_world[:, :3], quaternion=task_poses_in_world[:, 3:])
        world_tform_flattened_task = geometry.flatten_task(world_tform_task)
        task_tform_world: cuRoboPose = world_tform_flattened_task.inverse()

        optimal_pose_in_tasks: cuRoboPose = task_tform_world.multiply(optimal_pose_in_world.repeat(task_tform_world.batch))

        two_pi: float = 2*torch.pi

        yaw_angles: Tensor = geometry.extract_yaw_from_quaternions(optimal_pose_in_tasks.quaternion)
        yaw_angles = (yaw_angles + two_pi) % (two_pi)
        x_pos: Tensor = optimal_pose_in_tasks.position[:, 0]
        y_pos: Tensor = optimal_pose_in_tasks.position[:, 1]

        x_min, y_min = torch.amin(reference_poses_in_task.position[:, :2], dim=0)
        x_max, y_max = torch.amax(reference_poses_in_task.position[:, :2], dim=0)

        yaw_res: int = self.place_net_config.inverse_reachability.solution_resolution['yaw']
        x_res: int = self.place_net_config.inverse_reachability.solution_resolution['x']
        y_res: int = self.place_net_config.inverse_reachability.solution_resolution['y']

        yaw_indices: Tensor = torch.round(yaw_angles / (two_pi / (yaw_res-1)))
        x_indices = (x_res - 1) * ((x_pos - x_min) / (x_max - x_min))
        y_indices = (y_res - 1) * ((y_pos - y_min) / (y_max - y_min))

        valid_indices = (x_indices >= 0) & (x_indices < x_res) & (y_indices >= 0) & (y_indices < y_res)
        batch_indices = torch.arange(batch_size, dtype=int, device=device)
        valid_batch_indices = batch_indices[valid_indices]

        x_indices = x_indices[valid_indices].long()
        y_indices = y_indices[valid_indices].long()
        yaw_indices = yaw_indices[valid_indices].long()

        reachable_mask = valid_poses[valid_batch_indices, y_indices, x_indices, yaw_indices]
        return valid_batch_indices[reachable_mask]
    
    def pose_array_to_tensor(self, pose_array: PoseArray, target_frame: str, pose_link: str = '') -> Tensor:
        """
        Transform a pose array to a target frame and encode it into a PyTorch Tensor of shape (n, 7)
        """

        # First we check to see if the poses are in the desired frame, and if not we transform them
        needs_transform = pose_array.header.frame_id != target_frame
        if needs_transform: 
            self.get_logger().info(f'Transforming task frames from {pose_array.header.frame_id} to {target_frame}')
            poses = [
                self.tf_buffer.transform(
                    PoseStamped(pose=pose, header=pose_array.header),
                    target_frame,
                    timeout=rclpy.duration.Duration(seconds=3.0)
                ).pose for pose in pose_array.poses
            ]
        else:
            poses = pose_array.poses

        pose_curobo = place_net_conversions.poses_to_curobo(poses, self.place_net_config.model.device)

        # Then we check to see if the link the poses represents is the same as the ee_link. If it is not, then we
        # determine where the ee_link would be if the reported link is at these poses
        if (pose_link) and (pose_link != self.place_net_config.robot_config.robot.kinematics.kinematics_config.ee_link):
            link_tform_ee = self.tf_buffer.lookup_transform(
                target_frame=pose_link,
                source_frame=self.place_net_config.robot_config.robot.kinematics.kinematics_config.ee_link,
                time=rclpy.time.Time(seconds=0),
                timeout=rclpy.duration.Duration(seconds=3.0)
            )
            link_tform_ee_curobo = place_net_conversions.transform_to_curobo(link_tform_ee, self.place_net_config.model.device)
            pose_curobo = pose_curobo.multiply(link_tform_ee_curobo.repeat(pose_curobo.batch))

        return torch.cat([pose_curobo.position, pose_curobo.quaternion], dim=1)
    
    def pointcloud_to_tensor(self, pointcloud: PointCloud2, target_frame: str, filter_std_dev: float = 0.0) -> Tensor:
        """
        Encode the xyz fields of a pointcloud into a PyTorch Tensor and transform to a given frame
        """

        # Handle the case of an empty pointcloud
        if pointcloud.width == 0:
            return torch.tensor([], device=self.place_net_config.model.device)

        pointcloud_points = structured_to_unstructured(read_points(pointcloud, ['x', 'y', 'z'], skip_nans=True))
        if filter_std_dev > 0.0:
            pointcloud_open3d = open3d.geometry.PointCloud(points=pointcloud_points)
            pointcloud_open3d = pointcloud_open3d.remove_statistical_outlier(nb_neighbors=10, std_ratio=filter_std_dev)
            pointcloud_points = np.asarray(pointcloud_open3d.points)
        pointcloud_tensor = torch.tensor(pointcloud_points.copy(), device=self.place_net_config.model.device)

        if target_frame != pointcloud.header.frame_id:
            self.get_logger().info(f'Transforming pointcloud from {pointcloud.header.frame_id} to {target_frame}')
            transform = self.tf_buffer.lookup_transform(
                target_frame=target_frame, 
                source_frame=pointcloud.header.frame_id,
                time=rclpy.time.Time.from_msg(pointcloud.header.stamp),
                timeout=rclpy.duration.Duration(seconds=3.0)
            ).transform
            world_tform_pointcloud = place_net_conversions.transform_to_curobo(transform, self.place_net_config.model.device)
            pointcloud_tensor = world_tform_pointcloud.transform_points(pointcloud_tensor)

        return pointcloud_tensor

    def base_location_callback(self, req: QueryBaseLocation.Request, resp: QueryBaseLocation.Response):
        self.get_logger().info(' ')
        self.get_logger().info(f'[BaseLocationQuery]: Determining base pose to reach {len(req.end_effector_poses.poses)} task poses')
        
        if self.params.visualize:
            self.get_logger().info("Visualizing request now.")
            self.place_net_viz.visualize_query(req)

        # Make sure the task poses and pointclouds are represented in the same frame
        try:
            task_poses = self.pose_array_to_tensor(req.end_effector_poses, target_frame=self.params.world_frame, pose_link=req.pose_link)
            pointcloud_tensor = self.pointcloud_to_tensor(req.pointcloud, target_frame=self.params.world_frame, filter_std_dev=req.filter_std_dev)
        except Exception as e:
            self.get_logger().error(f'Caught error in base location callback: {e}')
            resp.success = False
            return resp

        # Get the output from the model
        t1 = time.perf_counter()
        try:
            model_output = self.get_solution_tensor(task_poses, pointcloud_tensor, req.mode)
            pose_scores = self.pose_scorer.score_pose_array(model_output)
        except RuntimeError as e:
            self.get_logger().error(f'Caught error calculating solution: {e}')
            resp.success = False
            return resp
        t2 = time.perf_counter()
        model_run_time = t2 - t1
        self.get_logger().info(f'Request using {req.mode} took {model_run_time:.3f} seconds')

        # Create a score tensor which covers all poses
        master_grid = self.create_master_score_grid(task_poses)
        task_pose_max = task_poses[:, :2].max(dim=0)[0]
        task_pose_min = task_poses[:, :2].min(dim=0)[0]
        master_grid.translate((task_pose_max + task_pose_min)/2)

        # Place all solutions into a master grid
        t3 = time.perf_counter()
        self.populate_master_score_grid(master_grid, task_poses, pose_scores)
        relative_scores = master_grid.scores / master_grid.scores.max()
        master_grid.scores /= task_poses.size(0)
        t4 = time.perf_counter()
        master_grid_time = t4 - t3
        self.get_logger().info(f'Score grid population took {master_grid_time:.3f} seconds')

        # The robot is inverted here so ee is actually base_link
        model_base_link: str = self.place_net_config.robot_config.inverted_robot.kinematics.kinematics_config.ee_link

        # Transform poses to the requested base link frame
        if model_base_link != req.base_link:
            self.get_logger().info(f'Transforming calculated poses from native frame {model_base_link} to requested frame {req.base_link}')
            try:
                manipulation_tform_base_link_ros = self.tf_buffer.lookup_transform(
                    target_frame=model_base_link, 
                    source_frame=req.base_link,
                    time=rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=3.0)
                ).transform
            except LookupException as e:
                self.get_logger().warn(f'Unable to transform place_net results to requested link frame "{req.base_link}": {e}')
                resp.has_valid_pose = False
                return resp

            manipulation_tform_base_link_ros = place_net_conversions.transform_to_curobo(manipulation_tform_base_link_ros, self.place_net_config.model.device)
            base_link_poses = master_grid.poses.multiply(manipulation_tform_base_link_ros.repeat(master_grid.poses.batch))
        else:
            base_link_poses = master_grid.poses

        # === Populate the response === #

        # Some fields are only populated if there is a valid pose
        resp.has_valid_pose = torch.any(model_output).item()
        if resp.has_valid_pose:
            self.get_logger().info("There is a valid pose")

            t5 = time.perf_counter()
            _, best_pose_idx = self.pose_scorer.select_best_pose(master_grid.scores.unsqueeze(0), already_scored=True)
            t6 = time.perf_counter()
            best_pose_time = t6 - t5
            resp.query_time = model_run_time + master_grid_time + best_pose_time
            resp.optimal_base_pose_index = best_pose_idx.item()

            best_pose = base_link_poses[best_pose_idx]
            resp.optimal_base_pose.header.frame_id = self.params.world_frame
            resp.optimal_base_pose.header.stamp = self.get_clock().now().to_msg()
            resp.optimal_base_pose.pose = place_net_conversions.curobo_pose_to_pose_list(best_pose)[0]

            resp.optimal_score = master_grid.scores.flatten()[best_pose_idx].double().item()
            self.get_logger().info(f'Optimal score is {resp.optimal_score:.3f}')

            reachable_pose_indices = self.get_reachable_pose_indices(
                optimal_pose_in_world   = master_grid.poses[best_pose_idx],
                task_poses_in_world     = task_poses,
                reference_poses_in_task = self.base_poses_in_flattened_task_frame,
                valid_poses             = model_output
            )
            resp.valid_task_indices = reachable_pose_indices.flatten().cpu().numpy().tolist()
            self.get_logger().info(f'{len(resp.valid_task_indices)}/{len(req.end_effector_poses.poses)} poses are reachable from the optimal pose')
        else:
            resp.query_time = model_run_time + master_grid_time
            self.get_logger().info("There are no valid poses")

        # Report which task poses had no reachable base poses at all
        invalid_layer_mask = torch.logical_not(torch.any(model_output.flatten(start_dim=1), dim=1))
        invalid_layer_indices = torch.arange(0, len(req.end_effector_poses.poses), device=invalid_layer_mask.device)[invalid_layer_mask].int()
        resp.unreachable_task_indices = invalid_layer_indices.flatten().cpu().tolist()

        valid_pose_mask = master_grid.scores.bool()
        resp.valid_poses.header.frame_id = self.params.world_frame
        resp.valid_poses.poses = place_net_conversions.curobo_pose_to_pose_list(base_link_poses[valid_pose_mask.flatten()])
        resp.valid_pose_scores = master_grid.scores[valid_pose_mask].flatten().cpu().double().numpy().tolist()

        # === Visualize the output === #

        if self.params.visualize:
            self.get_logger().info(f'Visualizing final scores')
            self.place_net_viz.visualize_response(req, resp, base_link_poses, relative_scores, task_poses, self.params.world_frame)
            self.get_logger().info(f'Done')
            
            self.place_net_viz.visualize_task_pointclouds(task_poses, pointcloud_tensor, self.params.world_frame)

            if self.place_net_viz.model_output_pub.get_subscription_count() > 0:
                self.get_logger().info(f'Visualizing model output')
                model_output_thread = Thread(target=self.place_net_viz.visualize_model_output, args=(task_poses, model_output, self.base_poses_in_flattened_task_frame, self.params.world_frame))
                model_output_thread.start()

        resp.success = True
        self.get_logger().info('Base placement query completed successfully')
        return resp
    
    def get_reachable_indices_gt(self, req: QueryReachablePoses.Request, pointcloud_tensor_in_world: Tensor, task_poses_in_world: Tensor) -> Tensor:
        # Generate the environment model
        pointcloud_points = pointcloud_tensor_in_world.cpu().numpy()
        if len(pointcloud_points) > 0:
            world_mesh = Mesh.from_pointcloud(pointcloud=pointcloud_points, pitch=0.01)
            world_config = WorldConfig(mesh=[world_mesh])
        else:
            world_config = None

        # Generate an IK solver for this problem
        ik_solver_config = IKSolverConfig.load_from_robot_config(
            self.place_net_config.robot_config.robot,
            world_config,
            rotation_threshold=req.rotation_threshold,
            position_threshold=req.position_threshold,
            num_seeds=req.num_seeds,
            self_collision_check=req.check_self_collision,
            self_collision_opt=req.check_self_collision,
            tensor_args=TensorDeviceType(device=self.place_net_config.model.device),
            use_cuda_graph=True
        )
        ik_solver = IKSolver(ik_solver_config)

        # Solve the IK problem
        task_poses_in_world = cuRoboPose(position=task_poses_in_world[:, :3], quaternion=task_poses_in_world[:, 3:])
        self.get_logger().info(f'Solving IK problem for {task_poses_in_world.batch} poses')
        success, joint_states = solve_batched_ik(ik_solver, self.place_net_config.max_ik_count, task_poses_in_world)
        self.get_logger().info(f'IK solving complete. There were {torch.sum(success, dtype=int)} reachable poses')
        
        return torch.arange(end=task_poses_in_world.batch)[success]
    
    def reachable_poses_callback(self, req: QueryReachablePoses.Request, resp: QueryReachablePoses.Response) -> QueryReachablePoses.Response:
        self.get_logger().info(' ')
        self.get_logger().info(f'[ReachabilityQuery]: Determine which of {len(req.end_effector_poses.poses)} task poses can be reached')
        
        if self.params.visualize:
            self.get_logger().info("Visualizing request now.")
            self.place_net_viz.visualize_query(req)

        if req.mode not in ["model", "irm", "ground_truth"]:
            self.get_logger().error(f'Invalid mode "{req.mode}" for reachability query. Options are ["model", "irm", "ground_truth"]')
            resp.success = False
            return resp

        # Get the required transforms for the pointcloud and for the task poses
        world_frame: str = self.params.world_frame
        ref_frame:   str = req.link_pose.header.frame_id
        robot_link:  str = req.link_frame
        model_base:  str = self.place_net_config.robot_config.inverted_robot.kinematics.kinematics_config.ee_link
        try:
            world_tform_ref_stamped = self.tf_buffer.lookup_transform(
                world_frame,
                ref_frame,
                rclpy.time.Time(seconds=0),
                rclpy.duration.Duration(seconds=0.5)
            )
            robot_link_tform_model_base_stamped = self.tf_buffer.lookup_transform(
                robot_link,
                model_base,
                rclpy.time.Time(seconds=0),
                rclpy.duration.Duration(seconds=0.5)
            )
        except LookupException as e:
            self.get_logger().error(f'Cannot complete ReachablePoses query as one of the necessary transforms cannot be found: {e}')
            resp.success = False
            return resp
        
        # Given the supplied base link, determine the pose of the model base link in the world frame
        world_tform_ref = place_net_conversions.transform_to_curobo(world_tform_ref_stamped.transform, self.place_net_config.model.device)
        ref_tform_robot_link = place_net_conversions.pose_to_curobo(req.link_pose.pose, self.place_net_config.model.device)
        robot_link_tform_model_base = place_net_conversions.transform_to_curobo(robot_link_tform_model_base_stamped.transform, self.place_net_config.model.device)
        model_base_in_world = world_tform_ref.multiply(ref_tform_robot_link).multiply(robot_link_tform_model_base)
            
        # Make sure the task poses and pointclouds are represented in the same frame
        try:
            task_poses = self.pose_array_to_tensor(req.end_effector_poses, target_frame=self.params.world_frame, pose_link=req.pose_link)
            pointcloud_tensor = self.pointcloud_to_tensor(req.pointcloud, target_frame=self.params.world_frame, filter_std_dev=req.filter_std_dev)
        except Exception as e:
            self.get_logger().error(f'Caught error in reachability callback: {e}')
            resp.success = False
            return resp
        
        # Filter out task poses that are too far away to matter
        distances = (model_base_in_world.position.expand(task_poses.size(0), -1) - task_poses[:, :3]).norm(dim=1).squeeze()
        valid_pose_mask = distances <= self.place_net_config.task_geometry.max_radial_reach
        valid_pose_indices = torch.arange(end=distances.numel()).to(self.place_net_config.model.device)[valid_pose_mask]
        valid_poses = task_poses[valid_pose_indices, :]
        self.get_logger().info(f'Performing inference on {valid_poses.size(0)}/{len(req.end_effector_poses.poses)} poses inside the reachability sphere')

        # Get the output from the model
        t1 = time.perf_counter()
        if req.mode == "ground_truth":
            reachable_pose_indices = self.get_reachable_indices_gt(req, pointcloud_tensor, valid_poses)
        else:
            try:
                model_output = self.get_solution_tensor(valid_poses, pointcloud_tensor, req.mode)
            except RuntimeError as e:
                self.get_logger().error(f'Caught error calculating solution: {e}')
                resp.success = False
                return resp

            # Determine which task poses are reachable from the supplied pose
            reachable_pose_indices = self.get_reachable_pose_indices(model_base_in_world, valid_poses, self.base_poses_in_flattened_task_frame, model_output)
        t2 = time.perf_counter()
        self.get_logger().info(f'Request using {req.mode} took {t2 - t1:.3f} seconds')

        resp.valid_task_indices = valid_pose_indices[reachable_pose_indices].flatten().cpu().numpy().tolist()
        resp.success = True
        self.get_logger().info(f'We can reach {len(resp.valid_task_indices)}/{len(req.end_effector_poses.poses)} task poses')
    
        if self.params.visualize:
            self.get_logger().info(f'Visualizing final scores')
            self.place_net_viz.visualize_reachability_response(req, resp, task_poses, self.params.world_frame)
    
        return resp

def main():
    rclpy.init()
    place_net_server = PlaceNetServer()
    place_net_server.get_logger().info(f'PlaceNet server online, using cuda device {place_net_server.params.device}')
    rclpy.spin(place_net_server)

if __name__ == '__main__':
    main()