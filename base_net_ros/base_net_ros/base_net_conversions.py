import torch
import numpy as np
from geometry_msgs.msg import Point, Quaternion, Transform, TransformStamped, PoseArray
from geometry_msgs.msg import Pose as RosPose, PoseStamped as RosPoseStamped
from curobo.types.math import Pose as cuRoboPose
from tf_transformations import quaternion_matrix

def tensor_to_point(point_tensor: torch.Tensor) -> Point:
    point_tensor = point_tensor.cpu().numpy().astype(float)
    return Point(x=point_tensor[0], y=point_tensor[1], z=point_tensor[2])

def tensor_to_quat(quat_tensor: torch.Tensor) -> Quaternion:
    quat_tensor = quat_tensor.cpu().numpy().astype(float)
    return Point(w=quat_tensor[0], x=quat_tensor[1], y=quat_tensor[2], z=quat_tensor[3])

def pose_tensor_to_pose_list(pose_tensor: torch.Tensor) -> list[RosPose]:
    pose_tensor = pose_tensor.cpu().numpy().astype(float)
    return [
        RosPose(
            position=Point(x=t[0], y=t[1], z=t[2]), 
            orientation=Quaternion(w=t[3], x=t[4], y=t[5], z=t[6])
        )
        for t in pose_tensor
    ]

def curobo_pose_to_pose_list(curobo_pose: cuRoboPose) -> list[RosPose]:
    return pose_tensor_to_pose_list(torch.cat([curobo_pose.position, curobo_pose.quaternion], dim=-1))

def poses_to_curobo(pose_list: list[RosPose] | PoseArray, device) -> cuRoboPose:
    if isinstance(pose_list, PoseArray):
        pose_list = pose_list.poses

    position_tensor = torch.tensor([[pose.position.x, pose.position.y, pose.position.z] for pose in pose_list], device=device, dtype=torch.float32)
    quaternion_tensor = torch.tensor([[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] for pose in pose_list], device=device, dtype=torch.float32)
    return cuRoboPose(position_tensor, quaternion_tensor)

def transform_to_curobo(transform: Transform, device) -> cuRoboPose:
    position = torch.tensor([transform.translation.x, transform.translation.y, transform.translation.z], device=device)
    quaternion = torch.tensor([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z], device=device)
    return cuRoboPose(position, quaternion)

def pose_to_curobo(pose: RosPose, device) -> cuRoboPose:
    position = torch.tensor([pose.position.x, pose.position.y, pose.position.z], device=device)
    quaternion = torch.tensor([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z], device=device)
    return cuRoboPose(position, quaternion)

def transform_to_matrix(tform: Transform | TransformStamped) -> np.ndarray:
    if isinstance(tform, TransformStamped):
        tform = tform.transform

    rot = tform.rotation
    tran = tform.translation
    transform_mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    transform_mat[:3, 3] = np.array([tran.x, tran.y, tran.z])
    return transform_mat

def pose_to_matrix(pose: RosPose | RosPoseStamped) -> np.ndarray:
    if isinstance(pose, RosPoseStamped):
        pose = pose.pose
        
    quat = pose.orientation
    pos = pose.position
    transform_mat = quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    transform_mat[:3, 3] = np.array([pos.x, pos.y, pos.z])
    return transform_mat