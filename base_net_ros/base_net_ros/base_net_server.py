import os
import time
import math
import torch

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy, create_cloud_xyz32

from curobo.types.math import Pose as cuRoboPose
from base_net.models.base_net import BaseNet
from base_net.utils.base_net_config import BaseNetConfig
from base_net_msgs.srv import QueryBaseLocation
from base_net.utils import geometry, pose_scorer

from .base_net_visualizer import BaseNetVisualizer

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
        self.base_net_config = BaseNetConfig.from_yaml_file(os.path.join(base_path, 'config.yaml'), load_solutions=False, load_tasks=False, device=device_param)
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

    def base_location_callback(self, req: QueryBaseLocation.Request, resp: QueryBaseLocation.Response):
        self.base_net_viz.visualize_query(req)
        batch_size = len(req.end_effector_poses.poses)

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
        pointcloud_list = [torch.tensor(pointcloud_points, device=self.base_net_config.model.device)]*batch_size

        # Get the output from the model
        t1 = time.perf_counter()
        model_output = torch.zeros(batch_size, self.base_net_config.inverse_reachability.solution_resolution['y'], self.base_net_config.inverse_reachability.solution_resolution['x'], self.base_net_config.inverse_reachability.solution_resolution['yaw'], dtype=bool)
        with torch.no_grad():
            for index_start in range(0, batch_size, self.max_batch_size):
                index_end = min(index_start + self.max_batch_size, batch_size)
                pointcloud_slice = pointcloud_list[index_start:index_end]
                task_slice = task_poses[index_start:index_end]
                logits = self.base_net_model(pointcloud_slice, task_slice)
                model_output[index_start:index_end] = torch.sigmoid(logits) >= 0.5
        model_output = model_output.to(self.base_net_config.model.device)
        t2 = time.perf_counter()
        print(f'Model forward pass took {t2 - t1} seconds')

        pose_scores = self.pose_scorer.score_pose_array(model_output)

        self.base_net_viz.visualize_model_output(task_poses, model_output, self.base_poses_in_flattened_task_frame, pointcloud_frame)
        self.base_net_viz.visualize_task_pointclouds(task_poses, pointcloud_list[0], pointcloud_frame)

        """TODO: Move this to its own function"""
        # Create a score tensor which covers all poses
        t3 = time.perf_counter()
        min_x, min_y = torch.amin(task_poses[:, :2], dim=0)
        max_x, max_y = torch.amax(task_poses[:, :2], dim=0)

        x_cell_size: float = 2*self.base_net_config.task_geometry.max_radial_reach / (self.base_net_config.inverse_reachability.solution_resolution['x'] - 1)
        y_cell_size: float = 2*self.base_net_config.task_geometry.max_radial_reach / (self.base_net_config.inverse_reachability.solution_resolution['y'] - 1)

        x_range: float = max_x - min_x + 2*self.base_net_config.task_geometry.max_radial_reach
        y_range: float = max_y - min_y + 2*self.base_net_config.task_geometry.max_radial_reach
        x_res: int = math.floor(x_range / x_cell_size) + 1
        y_res: int = math.floor(y_range / y_cell_size) + 1
        yaw_res: int = self.base_net_config.inverse_reachability.solution_resolution['yaw']

        master_grid_poses = geometry.load_base_pose_array(x_range/2, y_range/2, x_res, y_res, yaw_res, device=self.base_net_config.model.device)
        task_pose_max = task_poses[:, :2].max(dim=0)[0]
        task_pose_min = task_poses[:, :2].min(dim=0)[0]
        master_grid_poses.position[:, :2] += (task_pose_max + task_pose_min)/2
        min_grid_x, min_grid_y = torch.amin(master_grid_poses.position[:, :2], dim=0)
        score_tensor = torch.zeros((y_res, x_res, yaw_res), dtype=torch.float, device=self.base_net_config.model.device)
        t4 = time.perf_counter()
        print(f'Score grid creation took {t4 - t3} seconds')

        # Place all solutions into a master grid
        t5 = time.perf_counter()
        yaw_angles = geometry.extract_yaw_from_quaternions(task_poses[:, 3:])
        grid_lower_bound = torch.tensor([min_grid_x, min_grid_y], device=self.base_net_config.model.device)
        grid_extents = torch.tensor([x_range, y_range], device=self.base_net_config.model.device)
        grid_size = torch.tensor([x_res, y_res], device=self.base_net_config.model.device)
        for task_pose, yaw_angle, model_output_layer in zip(task_poses, yaw_angles, pose_scores):
            # Transform the results grid to this tasks base pose
            task_pose_curobo = cuRoboPose(position=task_pose[:3], quaternion=task_pose[3:])
            world_tform_flattened_task = geometry.flatten_task(task_pose_curobo)
            base_poses_in_world: cuRoboPose = world_tform_flattened_task.repeat(self.base_poses_in_flattened_task_frame.batch).multiply(self.base_poses_in_flattened_task_frame)

            # We only need to update entries that have reachable poses
            valid_model_indices = model_output_layer.view(-1, yaw_res).sum(dim=1, dtype=bool)

            # Calculate the indices into the yaw angles
            yaw_index_offset: int = round(yaw_angle.item() / (2*math.pi / yaw_res))
            yaw_indices = torch.arange(yaw_res, device=self.base_net_config.model.device) + yaw_index_offset
            yaw_indices = torch.remainder(yaw_indices, yaw_res)
            yaw_indices = yaw_indices.long()

            # Calculate the indices into the positions
            xy_positions = base_poses_in_world.position[:, :2][::yaw_res]
            offsets = ((xy_positions - grid_lower_bound)) / grid_extents
            grid_indices = torch.round(offsets * grid_size)
            valid_indices = ((grid_indices >= 0) & (grid_indices < grid_size)).prod(dim=1, dtype=bool)
            valid_indices = valid_indices & valid_model_indices
            grid_indices = grid_indices[valid_indices]
            grid_indices = grid_indices[:, [1, 0]] # grid is arranged (y, x, yaw)
            grid_indices = grid_indices.long()

            # Interleave the position and yaw indices
            yaw_indices_interleaved = yaw_indices.view(1, -1, 1).expand(grid_indices.size(0), -1, 1)
            grid_indices_interleaved = grid_indices.unsqueeze(1).expand(-1, yaw_res, 2)
            layer_indices = torch.concatenate([grid_indices_interleaved, yaw_indices_interleaved], dim=-1)
            layer_indices = layer_indices.view(-1, 3)

            # Assign to the score tensor
            y_indices = layer_indices[:, 0]
            x_indices = layer_indices[:, 1]
            t_indices = layer_indices[:, 2]
            score_tensor[y_indices, x_indices, t_indices] += model_output_layer.view(-1, yaw_res)[valid_indices, :].flatten().float()
        score_tensor /= batch_size
        t6 = time.perf_counter()
        print(f'Score grid population took {t6 - t5} seconds')
            
        """TODO: Score the master grid and select the highest score"""

        resp.has_valid_pose = torch.any(model_output).item()
        print(f'Has valid pose: {resp.has_valid_pose}')

        pose_scores = self.pose_scorer.score_pose_array(model_output.cpu()).squeeze(0)
        self.base_net_viz.visualize_response(resp, master_grid_poses, score_tensor, pointcloud_frame)
        return resp

def main():
    rclpy.init()
    base_net_server = BaseNetServer()
    base_net_server.get_logger().info('BaseNet server online')
    rclpy.spin(base_net_server)

if __name__ == '__main__':
    main()