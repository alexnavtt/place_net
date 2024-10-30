import os
import torch

import rclpy
import rclpy.duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped
from sensor_msgs_py.point_cloud2 import read_points

from base_net.models.base_net import BaseNet
from base_net.utils.base_net_config import BaseNetConfig
from base_net_msgs.srv import QueryBaseLocation
from base_net.utils import geometry

class BaseNetServer(Node):
    def __init__(self):
        super(BaseNetServer).__init__('base_net_server')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Load the model from the checkpoint path
        checkpoint_path: str = self.declare_parameter('checkpoint_path').value
        device_param: str = self.declare_parameter('device').value
        self.max_batch_size: int = self.declare_parameter('max_batch_size').value

        base_path, _ = os.path.split(checkpoint_path)
        self.base_net_config = BaseNetConfig.from_yaml_file(os.path.join(base_path, 'config.yaml'), load_solutions=False, load_tasks=False, device=device_param)
        self.base_net_model = BaseNet(self.base_net_config)

        checkpoint_config = torch.load(checkpoint_path, map_location=self.base_net_config.model.device)
        self.base_net_model.load_state_dict(checkpoint_config['base_net_model'])
        self.base_net_model.eval()

        # Load the model geometry
        _, self.base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
            reach_radius=self.base_net_config.task_geometry.max_radial_reach,
            x_res=self.base_net_config.inverse_reachability.solution_resolution['x'],
            y_res=self.base_net_config.inverse_reachability.solution_resolution['y'],
            yaw_res=self.base_net_config.inverse_reachability.solution_resolution['yaw'],
            device=self.base_net_config.model.device
        )

        # Start up the ROS service
        self.base_location_server = self.create_service(QueryBaseLocation, '~/query_base_location', self.base_location_callback)

    def base_location_callback(self, req: QueryBaseLocation.Request) -> QueryBaseLocation.Response:
        batch_size = len(req.end_effector_poses.poses)

        # Make sure the task poses and pointclouds are represented in the same frame
        pointcloud_frame: str = req.pointcloud.header.frame_id

        task_poses = torch.zeros((len(req.end_effector_poses.poses), 7))
        for idx, pose in enumerate(req.end_effector_poses.poses):
            pose_stamped = PoseStamped(pose=pose, header=req.end_effector_poses.header)
            transfomed_pose: PoseStamped = self.tf_buffer.transform(pose_stamped, pointcloud_frame, timeout=rclpy.duration.Duration(1.0))
            task_poses[idx, :3] = [transfomed_pose.pose.position.x, transfomed_pose.pose.position.y, transfomed_pose.pose.position.z]
            task_poses[idx, 3:] = [transfomed_pose.pose.orientation.w, transfomed_pose.pose.orientation.x, transfomed_pose.pose.orientation.y, transfomed_pose.pose.orientation.z]

        # Encode the pointcloud into a tensor
        pointcloud_points = read_points(req.pointcloud, 'xyz', skip_nans=True)
        pointcloud_list = [torch.tensor(pointcloud_points)]*batch_size

        # Get the output from the model
        model_output = torch.zeros(batch_size, self.base_net_config.inverse_reachability.solution_resolution['y'], self.base_net_config.inverse_reachability.solution_resolution['x'], self.base_net_config.inverse_reachability.solution_resolution['yaw'])
        for index_start in range(0, batch_size+self.max_batch_size, self.max_batch_size):
            index_end = min(index_start + self.max_batch_size, batch_size)
            pointcloud_slice = pointcloud_list[index_start:index_end]
            task_slice = task_poses[index_start:index_end]
            model_output[index_start:index_end] = self.base_net_model.forward(pointcloud_slice, task_slice)

        """TODO: Place all solutions into a master grid"""
        """TODO: Move this to its own function"""
        min_x = torch.min(task_poses[:, 0])
        max_x = torch.max(task_poses[:, 0])
        min_y = torch.min(task_poses[:, 1])
        max_y = torch.max(task_poses[:, 1])
        min_z = torch.min(task_poses[:, 2])
        max_z = torch.max(task_poses[:, 2])

        x_res = 2*self.base_net_config.task_geometry.max_radial_reach / (self.base_net_config.inverse_reachability.solution_resolution['x'] - 1)
        y_res = 2*self.base_net_config.task_geometry.max_radial_reach / (self.base_net_config.inverse_reachability.solution_resolution['y'] - 1)
        yaw_res = 2*torch.pi / self.base_net_config.inverse_reachability.solution_resolution['yaw']

        """TODO: Score the master grid and select the highest score"""

def main():
    rclpy.init()
    base_net_server = BaseNetServer('base_net_server')
    rclpy.spin(base_net_server)

if __name__ == '__main__':
    main()