import copy
import torch
from tqdm import tqdm

from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.node import Node
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from geometry_msgs.msg import PoseStamped, PoseArray, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

from curobo.types.math import Pose as cuRoboPose

from base_net_msgs.srv import QueryBaseLocation
from base_net.utils.base_net_config import BaseNetConfig
from base_net.utils import geometry

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
        self.response_unreachable_task_pub = ros_node.create_publisher(PoseArray  , '~/response/unreachable_tasks'    , latching_qos)
        self.response_reachable_task_pub   = ros_node.create_publisher(PoseArray  , '~/response/reachable_tasks'      , latching_qos)
        self.response_aggregate_scores_pub = ros_node.create_publisher(MarkerArray, '~/response/aggregate_pose_scores', latching_qos)

        # Ground truth visualizations
        self.ground_truth_valid_pub = ros_node.create_publisher(PoseArray, '~/ground_truth/valid_poses', latching_qos)
        self.ground_truth_task_pub  = ros_node.create_publisher(PoseArray, '~/ground_truth/task_poses', latching_qos)

        # Debug visualization
        self.model_output_pub    = ros_node.create_publisher(MarkerArray, '~/model_output'   , latching_qos)
        self.points_in_range_pub = ros_node.create_publisher(PointCloud2, '~/points_in_range', latching_qos)
        self.gt_points_pub       = ros_node.create_publisher(PointCloud2, '~/ground_truth_points', latching_qos)

    def visualize_query(self, req: QueryBaseLocation.Request) -> None:
        self.query_task_pub.publish(req.end_effector_poses)
        self.query_pointcloud_pub.publish(req.pointcloud)

    def pose_array_to_marker(self, pose_array: cuRoboPose, scores: torch.Tensor, frame_id: str) -> Marker:
        arrow_marker = Marker()
        arrow_marker.action = Marker.ADD
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = self.ros_node.get_clock().now().to_msg()
        arrow_marker.scale = Vector3(x=0.005)
        arrow_marker.type = Marker.LINE_LIST

        qw, qx, qy, qz = pose_array.quaternion.split([1, 1, 1, 1], dim=-1)
        fx = 1 - 2*(qy**2 + qz**2)
        fy = 2*(qx*qy + qw*qz)
        fz = 2*(qx*qz - qw*qy)
        forward_vectors = torch.cat([fx, fy, fz], dim=1)

        start_points = pose_array.position
        end_points = start_points + 0.05*forward_vectors
        arrow_points = torch.empty(2*pose_array.batch, 3, dtype=float)
        arrow_points[::2, :]  = start_points
        arrow_points[1::2, :] = end_points

        scores = scores.flatten().float()
        colors = torch.zeros(2*scores.size(0), 4, dtype=float)
        colors[::2, 0] = 1 - scores
        colors[::2, 1] = scores
        colors[::2, 2] = 0.5 + 0.5 * scores
        colors[1::2] = colors[::2]

        arrow_marker.points.extend([Point(x=pt[0], y=pt[1], z=pt[2]) for pt in arrow_points.cpu().numpy()])
        arrow_marker.colors.extend([ColorRGBA(r=cl[0], g=cl[1], a=cl[2]) for cl in colors.cpu().numpy()])

        return arrow_marker

    def visualize_response(self, req: QueryBaseLocation.Request, resp: QueryBaseLocation.Response, final_base_grid: cuRoboPose, scored_grid: torch.Tensor, frame_id: str) -> None:
        if resp.has_valid_pose:
            self.response_optimal_pub.publish(resp.optimal_base_pose)
        self.response_valid_pub.publish(resp.valid_poses)

        valid_task_pose_array = PoseArray()
        valid_task_pose_array.header = req.end_effector_poses.header
        valid_task_pose_array.poses = [req.end_effector_poses.poses[idx] for idx in resp.valid_task_indices]
        self.response_reachable_task_pub.publish(valid_task_pose_array)

        unreachable_task_pose_array = PoseArray()
        unreachable_task_pose_array.header = req.end_effector_poses.header
        unreachable_task_pose_array.poses = [req.end_effector_poses.poses[idx] for idx in resp.unreachable_task_indices]
        self.response_unreachable_task_pub.publish(unreachable_task_pose_array)

        final_grid_marker = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.id = -1
        final_grid_marker.markers.append(delete_marker)

        arrow_marker = self.pose_array_to_marker(final_base_grid, scored_grid, frame_id)

        final_grid_marker.markers.append(copy.deepcopy(arrow_marker))
        self.response_aggregate_scores_pub.publish(final_grid_marker)

    def visualize_model_output(self, tasks: torch.Tensor, model_output: torch.Tensor, base_pose_array: cuRoboPose, frame_id: str):
        markers = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.id = -1
        markers.markers.append(delete_marker)

        tasks = tasks.to(self.base_net_config.model.device)
        tasks[:, 2] = self.base_net_config.task_geometry.base_link_elevation
        model_output = model_output.cpu()

        # Only show a loading bar for larger batch sizes
        wrapper = tqdm if tasks.size(0) > 10 else lambda x, total: x

        for idx, (output_layer, task) in wrapper(enumerate(zip(model_output, tasks)), total=tasks.size(0)):
            task_pose = cuRoboPose(position=task[:3], quaternion=task[3:])
            world_tform_flattened_task = geometry.flatten_task(task_pose)
            base_pose_in_world: cuRoboPose = world_tform_flattened_task.repeat(base_pose_array.batch).multiply(base_pose_array)

            new_arrows = self.pose_array_to_marker(base_pose_in_world, output_layer, frame_id)
            new_arrows.id = idx
            markers.markers.append(new_arrows)

        self.model_output_pub.publish(markers)

    def visualize_task_pointclouds(self, tasks: torch.Tensor, pointcloud: torch.Tensor, frame_id: str):
        if pointcloud.numel() == 0:
            return

        point_mask = torch.zeros(pointcloud.size(0), dtype=bool, device=pointcloud.device)
        for task in tasks:
            point_mask |= (pointcloud[:, :2] - task[:2]).norm(dim=-1) < self.base_net_config.task_geometry.max_pointcloud_radius
        point_mask &= pointcloud[:, 2] > self.base_net_config.task_geometry.min_pointcloud_elevation
        point_mask &= pointcloud[:, 2] < self.base_net_config.task_geometry.max_pointcloud_elevation

        valid_points = pointcloud[point_mask]
        valid_points_ros = create_cloud_xyz32(header=Header(frame_id=frame_id, stamp=self.ros_node.get_clock().now().to_msg()), points=valid_points.cpu().numpy())
        self.points_in_range_pub.publish(valid_points_ros)