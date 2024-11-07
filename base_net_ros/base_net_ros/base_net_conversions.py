import torch
from geometry_msgs.msg import Point, Quaternion, Transform
from geometry_msgs.msg import Pose as RosPose
from curobo.types.math import Pose as cuRoboPose

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

def transform_to_curobo(transform: Transform, device) -> cuRoboPose:
    position = torch.tensor([transform.translation.x, transform.translation.y, transform.translation.z], device=device)
    quaternion = torch.tensor([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z], device=device)
    return cuRoboPose(position, quaternion)