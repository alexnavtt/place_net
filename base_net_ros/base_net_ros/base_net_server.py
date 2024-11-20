import os
import time
import math
import torch
from torch import Tensor
from threading import Thread

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped
from sensor_msgs_py.point_cloud2 import read_points_numpy

from curobo.types.math import Pose as cuRoboPose
from base_net.models.base_net import BaseNet
from base_net.utils.base_net_config import BaseNetConfig
from base_net_msgs.srv import QueryBaseLocation
from base_net.utils import geometry, pose_scorer

from .base_net_visualizer import BaseNetVisualizer
from . import base_net_conversions

class PoseGrid:
    def __init__(self, x_range: float, y_range: float, x_res: int, y_res: int, yaw_res: int, device):
        self.x_range = x_range
        self.y_range = y_range
        self.x_res = x_res
        self.y_res = y_res
        self.yaw_res = yaw_res
        self.device = device

        self.poses = geometry.load_base_pose_array(x_range/2, y_range/2, x_res, y_res, yaw_res, device=device)
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

class BaseNetServer(Node):
    def __init__(self):
        super().__init__(node_name='base_net_server')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Load the model from the checkpoint path
        checkpoint_path: str = self.declare_parameter('checkpoint_path').value
        device_param: str | int | None = self.declare_parameter('device').value
        self.max_batch_size: int = self.declare_parameter('max_batch_size').value

        base_path, _ = os.path.split(checkpoint_path)
        self.base_net_config = BaseNetConfig.from_yaml_file(os.path.join(base_path, 'config.yaml'), load_pointclouds=False, load_solutions=False, load_tasks=False, device=device_param)
        self.base_net_model = BaseNet(self.base_net_config)
        self.pose_scorer = pose_scorer.PoseScorer()

        checkpoint_config = torch.load(checkpoint_path, map_location=self.base_net_config.model.device)
        self.base_net_model.load_state_dict(checkpoint_config['base_net_model'])
        self.base_net_model.eval()

        # Load the model geometry
        self.base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
            half_x_range=self.base_net_config.task_geometry.max_radial_reach,
            half_y_range=self.base_net_config.task_geometry.max_radial_reach,
            x_res=self.base_net_config.inverse_reachability.solution_resolution['x'],
            y_res=self.base_net_config.inverse_reachability.solution_resolution['y'],
            yaw_res=self.base_net_config.inverse_reachability.solution_resolution['yaw'],
            device=self.base_net_config.model.device
        )
        self.base_net_viz = BaseNetVisualizer(self, self.base_net_config)

        # Start up the ROS service
        self.base_location_server = self.create_service(QueryBaseLocation, '~/query_base_location', self.base_location_callback)

    def run_model(self, task_poses: Tensor, pointcloud: Tensor) -> Tensor:
        batch_size = task_poses.size(0)
        pointcloud_list = [pointcloud]*batch_size

        model_output = torch.zeros(
            batch_size, 
            self.base_net_config.inverse_reachability.solution_resolution['y'],
            self.base_net_config.inverse_reachability.solution_resolution['x'],
            self.base_net_config.inverse_reachability.solution_resolution['yaw'], 
            dtype=bool,
            device=self.base_net_config.model.device
        )
        
        with torch.no_grad():
            for index_start in range(0, batch_size, self.max_batch_size):
                index_end = min(index_start + self.max_batch_size, batch_size)
                pointcloud_slice = pointcloud_list[index_start:index_end]
                task_slice = task_poses[index_start:index_end]
                logits = self.base_net_model(pointcloud_slice, task_slice)
                model_output[index_start:index_end] = torch.sigmoid(logits) >= 0.5

        return model_output
    
    def create_master_score_grid(self, task_poses: Tensor) -> PoseGrid:
        min_x, min_y = torch.amin(task_poses[:, :2], dim=0)
        max_x, max_y = torch.amax(task_poses[:, :2], dim=0)

        x_cell_size: float = 2*self.base_net_config.task_geometry.max_radial_reach / (self.base_net_config.inverse_reachability.solution_resolution['x'] - 1)
        y_cell_size: float = 2*self.base_net_config.task_geometry.max_radial_reach / (self.base_net_config.inverse_reachability.solution_resolution['y'] - 1)

        x_range: float = max_x - min_x + 2*self.base_net_config.task_geometry.max_radial_reach
        y_range: float = max_y - min_y + 2*self.base_net_config.task_geometry.max_radial_reach
        x_res: int = math.floor(x_range / x_cell_size) + 1
        y_res: int = math.floor(y_range / y_cell_size) + 1
        yaw_res: int = self.base_net_config.inverse_reachability.solution_resolution['yaw']

        return PoseGrid(x_range, y_range, x_res, y_res, yaw_res, self.base_net_config.model.device)
    
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
            yaw_indices = torch.arange(yaw_res, device=self.base_net_config.model.device) + yaw_index_offset
            yaw_indices = torch.remainder(yaw_indices, yaw_res)
            yaw_indices = yaw_indices.long()

            # Calculate the indices into the positions
            xy_positions = base_poses_in_world.position[:, :2][::yaw_res]
            offsets = ((xy_positions - master_grid.lower_bound)) / master_grid.extent
            float_grid_indices = offsets * master_grid.grid_size
            grid_indices = torch.floor(float_grid_indices)

            # Calculate the offsets from the nearest grid cells
            fractional_offsets = torch.frac(float_grid_indices)
            for grid_offset in torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=self.base_net_config.model.device):
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
            - optimal_pose_in_world: The chosen base pose for the robot. Shape (1)
            - task_poses_in_world: The task poses for which we have calculated reachability. Shape (B, 7)
            - reference_poses_in_task: The array of sample poses reachability was calculated for, defined in the flattened task frame. Shape (ny*nx*ntheta)
            - valid_poses: The boolean model output indicating which poses can be reached. Shape (B, ny, nx, ntheta) 
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

        yaw_res: int = self.base_net_config.inverse_reachability.solution_resolution['yaw']
        x_res: int = self.base_net_config.inverse_reachability.solution_resolution['x']
        y_res: int = self.base_net_config.inverse_reachability.solution_resolution['y']

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

    def base_location_callback(self, req: QueryBaseLocation.Request, resp: QueryBaseLocation.Response):
        self.get_logger().info("Received base location request. Visualizing request now.")
        self.base_net_viz.visualize_query(req)

        # Make sure the task poses and pointclouds are represented in the same frame
        pointcloud_frame: str = req.pointcloud.header.frame_id

        task_poses = torch.zeros((len(req.end_effector_poses.poses), 7))
        for idx, pose in enumerate(req.end_effector_poses.poses):
            pose_stamped = PoseStamped(pose=pose, header=req.end_effector_poses.header)
            transfomed_pose: PoseStamped = self.tf_buffer.transform(pose_stamped, pointcloud_frame, timeout=rclpy.duration.Duration(seconds=1.0))
            task_poses[idx, :3] = torch.tensor([transfomed_pose.pose.position.x, transfomed_pose.pose.position.y, transfomed_pose.pose.position.z])
            task_poses[idx, 3:] = torch.tensor([transfomed_pose.pose.orientation.w, transfomed_pose.pose.orientation.x, transfomed_pose.pose.orientation.y, transfomed_pose.pose.orientation.z])
        task_poses = task_poses.to(self.base_net_config.model.device)

        # Encode the pointcloud into a tensor
        pointcloud_points = read_points_numpy(req.pointcloud, ['x', 'y', 'z'], skip_nans=True)
        pointcloud_tensor = torch.tensor(pointcloud_points, device=self.base_net_config.model.device)

        # Get the output from the model
        t1 = time.perf_counter()
        model_output = self.run_model(task_poses, pointcloud_tensor)
        pose_scores = self.pose_scorer.score_pose_array(model_output)
        t2 = time.perf_counter()
        self.get_logger().info(f'Model forward pass took {t2 - t1:.3f} seconds')

        # Create a score tensor which covers all poses
        master_grid = self.create_master_score_grid(task_poses)
        task_pose_max = task_poses[:, :2].max(dim=0)[0]
        task_pose_min = task_poses[:, :2].min(dim=0)[0]
        master_grid.translate((task_pose_max + task_pose_min)/2)

        # Place all solutions into a master grid
        t5 = time.perf_counter()
        self.populate_master_score_grid(master_grid, task_poses, pose_scores)
        relative_scores = master_grid.scores / master_grid.scores.max()
        master_grid.scores /= task_poses.size(0)
        t6 = time.perf_counter()
        self.get_logger().info(f'Score grid population took {t6 - t5:.3f} seconds')

        resp.has_valid_pose = torch.any(model_output).item()

        if resp.has_valid_pose:
            self.get_logger().info("There is a valid pose")

            # Transform poses to the requested base link frame
            self.get_logger().info(f'Transforming calculated poses from native frame {self.base_net_config.robot.kinematics.kinematics_config.ee_link} to requested frame {req.base_link}')
            try:
                manipulation_tform_base_link_ros = self.tf_buffer.lookup_transform(
                    target_frame=self.base_net_config.robot.kinematics.kinematics_config.ee_link, # The robot is inverted here so ee is actually base_link
                    source_frame=req.base_link,
                    time=rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.25)
                ).transform
            except LookupException as e:
                self.get_logger().warn(f'Unable to transform base_net results to requested link frame "{req.base_link}": {e}')
                resp.has_valid_pose = False
                return resp

            manipulation_tform_base_link_ros: cuRoboPose = base_net_conversions.transform_to_curobo(manipulation_tform_base_link_ros, self.base_net_config.model.device)

            base_link_poses = master_grid.poses.multiply(manipulation_tform_base_link_ros.repeat(master_grid.poses.batch))
            base_link_poses.position[:, 2] = 0
            _, best_pose_idx = self.pose_scorer.select_best_pose(master_grid.scores.unsqueeze(0), already_scored=True)
            best_pose = base_link_poses[best_pose_idx]

            resp.optimal_base_pose.header.frame_id = pointcloud_frame
            resp.optimal_base_pose.header.stamp = self.get_clock().now().to_msg()
            resp.optimal_base_pose.pose = base_net_conversions.curobo_pose_to_pose_list(best_pose)[0]

            resp.optimal_score = master_grid.scores.flatten()[best_pose_idx].double().item()
            self.get_logger().info(f'Optimal score is {resp.optimal_score:.3f}')

            reachable_pose_indices = self.get_reachable_pose_indices(
                optimal_pose_in_world   = master_grid.poses[best_pose_idx],
                task_poses_in_world     = task_poses,
                reference_poses_in_task = self.base_poses_in_flattened_task_frame,
                valid_poses             = model_output
            )
            resp.valid_task_indices = reachable_pose_indices.flatten().cpu().numpy().tolist()

            valid_pose_mask = master_grid.scores.bool()
            resp.valid_poses.header = resp.optimal_base_pose.header
            resp.valid_poses.poses = base_net_conversions.curobo_pose_to_pose_list(base_link_poses[valid_pose_mask.flatten()])
            resp.valid_pose_scores = master_grid.scores[valid_pose_mask].flatten().cpu().double().numpy().tolist()

            self.get_logger().info(f'Visualizing final scores')
            self.base_net_viz.visualize_response(req, resp, base_link_poses, relative_scores, pointcloud_frame)
            self.get_logger().info(f'Done')
        else:
            self.get_logger().info("There are no valid poses")
        
        self.get_logger().info(f'Visualizing model output')
        self.base_net_viz.visualize_task_pointclouds(task_poses, pointcloud_tensor, pointcloud_frame)
        model_output_thread = Thread(target=self.base_net_viz.visualize_model_output, args=(task_poses, model_output, self.base_poses_in_flattened_task_frame, pointcloud_frame))
        model_output_thread.start()

        return resp

def main():
    rclpy.init()
    base_net_server = BaseNetServer()
    base_net_server.get_logger().info('BaseNet server online')
    rclpy.spin(base_net_server)

if __name__ == '__main__':
    main()