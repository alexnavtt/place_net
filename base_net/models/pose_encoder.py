import torch
from torch import Tensor

class PoseEncoder(torch.nn.Module):
    def __init__(self):
        super(PoseEncoder, self).__init__()

        # Task embedding layers
        self.task_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=32),
            torch.nn.BatchNorm1d(num_features=32),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=32, out_features=1024),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU(),
        )

    def encode(self, poses: Tensor, max_height: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Given a tensor of input poses, provide an encoded feature vector for those poses, 
        as well as a transformation matrix which transforms points defined in the same
        frame as poses to a new frame which is invariant to irrelevant information such 
        as x, y, yaw values

        Input:
            poses - Tensor of shape (batch_size, 7) described as [x, y, z, qw, qx, qy, qz]
        Output: 
            transform - Tensor of shape (batch_size, 4, 4) which can be used to transform 
                        other geometric quantities from the global frame to the pose task
                        invariant frame
            adjusted_pose - Tensor of shape (batch_size, 3) representing a the z, pitch, roll 
                            encoding of the task pose in its task invariant frame
            encoding - Tensor of shape (batch_size, 4) representing the normalized encoding
                       of the adjusted pose with fields z, pitch, sin(roll), cos(roll)
        """
        batch_size = poses.size()[0]
        task_positions, task_orientations = poses.split([3, 4], dim=1)
        
        euler_angles_rpy = self.quaternion_to_euler_tensor(task_orientations)
        roll_angles = euler_angles_rpy[:, 0]
        pitch_angles = euler_angles_rpy[:, 1]
        yaw_angles = euler_angles_rpy[:, 2]

        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)
        task_rot_world = torch.zeros((batch_size, 2, 2), device=poses.device, requires_grad=False)
        task_rot_world[:, 0, 0] = cos_yaw
        task_rot_world[:, 0, 1] = -sin_yaw
        task_rot_world[:, 1, 0] = sin_yaw
        task_rot_world[:, 1, 1] = cos_yaw

        adjusted_pose = torch.stack([task_positions[:, 2], pitch_angles, roll_angles], dim=1)
        task_encoding = torch.stack([(task_positions[:, 2] - max_height/2)/max_height, pitch_angles/(torch.pi/2), torch.sin(roll_angles), torch.cos(roll_angles)], dim=1)
        return task_rot_world, adjusted_pose, task_encoding

    def forward(self, encoded_poses: Tensor) -> Tensor:
        return self.task_embedding(encoded_poses)

    def quaternion_to_euler_tensor(self, quaternions) -> Tensor:
        qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # Precompute terms for efficiency
        qwx = qw*qx
        qwy = qw*qy
        qwz = qw*qz
        qxy = qx*qy
        qxz = qx*qz
        qyz = qy*qz
        qxx = qx*qx
        qyy = qy*qy
        qzz = qz*qz

        # Yaw calculation
        sin_yaw_cos_pitch = 2*(qwz + qxy)
        cos_yaw_cos_pitch = 1 - 2*(qyy + qzz)
        yaw = torch.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

        # Pitch calculation
        sin_pitch = 2*(qwy - qxz)
        sin_pitch = sin_pitch.clamp(-1.0, 1.0) # avoid numerical errors
        pitch = torch.asin(sin_pitch)

        # Roll calculation
        sin_roll_cos_pitch = 2*(qwx + qyz)
        cos_roll_cos_pitch = 1 - 2*(qxx + qyy)
        roll = torch.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

        return torch.stack((roll, pitch, yaw), dim=1) 