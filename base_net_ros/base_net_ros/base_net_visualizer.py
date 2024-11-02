import copy
import torch

from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Vector3, Pose
from visualization_msgs.msg import Marker, MarkerArray

from curobo.types.math import Pose as cuRoboPose

from base_net_msgs.srv import QueryBaseLocation
from base_net.utils.base_net_config import BaseNetConfig

class BaseNetVisualizer:
    def __init__(self, ros_node: Node, base_net_config: BaseNetConfig):
        self.ros_node = ros_node
        self.base_net_config = base_net_config

        latching_qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        # Request visualization
        self.query_task_pub       = ros_node.create_publisher(PoseArray  , '~/query/task_poses', latching_qos)
        self.query_pointcloud_pub = ros_node.create_publisher(PointCloud2, '~/query/pointcloud', latching_qos)

        # Response visualization
        self.response_optimal_pub          = ros_node.create_publisher(PoseStamped, '~/response/optimal_pose'         , latching_qos)
        self.response_valid_pub            = ros_node.create_publisher(PoseArray  , '~/response/valid_poses'          , latching_qos)
        self.response_scores_pub           = ros_node.create_publisher(MarkerArray, '~/response/pose_scores'          , latching_qos)
        self.response_aggregate_scores_pub = ros_node.create_publisher(MarkerArray, '~/response/aggregate_pose_scores', latching_qos)

    def visualize_query(self, req: QueryBaseLocation.Request) -> None:
        self.query_task_pub.publish(req.end_effector_poses)
        self.query_pointcloud_pub.publish(req.pointcloud)

    def visualize_response(self, resp: QueryBaseLocation.Response, final_base_grid: cuRoboPose, scored_grid: torch.Tensor, frame_id: str) -> None:
        self.response_optimal_pub.publish(resp.optimal_base_pose)
        self.response_valid_pub.publish(resp.valid_poses)

        final_base_grid_tensor = torch.concatenate([final_base_grid.position, final_base_grid.quaternion, scored_grid.flatten().unsqueeze(1)], dim=1)
        final_grid_marker = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.id = -1
        final_grid_marker.markers.append(delete_marker)

        arrow_marker = Marker()
        arrow_marker.action = Marker.ADD
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = self.ros_node.get_clock().now().to_msg()
        arrow_marker.id = 0
        arrow_marker.scale = Vector3(x=0.05, y=0.005, z=0.01)
        arrow_marker.type = Marker.ARROW

        for pose_vec in final_base_grid_tensor:
            position, quaternion, score = torch.split(pose_vec, [3, 4, 1])

            new_pose = Pose()
            new_pose.position.x, new_pose.position.y, new_pose.position.z = position.cpu().numpy().astype(float)
            new_pose.orientation.w, new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z = quaternion.cpu().numpy().astype(float)
            arrow_marker.pose = new_pose

            arrow_marker.color.g = score.item()
            arrow_marker.color.r = 1 - score.item()
            arrow_marker.color.a = 0.5 +  0.5*score.item()
            final_grid_marker.markers.append(copy.deepcopy(arrow_marker))
            arrow_marker.id += 1

        self.response_aggregate_scores_pub.publish(final_grid_marker)