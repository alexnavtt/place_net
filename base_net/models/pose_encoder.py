import torch

class PoseEncoder(torch.nn.Module):
    def __init__(self, device):
        super(PoseEncoder, self).__init__()

        self.device = device

        # Task embedding layers
        self.task_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=32, device=device),
            torch.nn.BatchNorm1d(num_features=32, device=device),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=32, out_features=256, device=device),
            torch.nn.BatchNorm1d(num_features=256, device=device),
            torch.nn.ReLU()
        )

    def forward(self, poses: torch.Tensor, max_height: float) -> tuple[torch.Tensor, torch.Tensor]:
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
            embedding - Tensor of shape (batch_size, 256) representing a feature embedding
                        of the task pose in its task invariant frame
        """
        batch_size = poses.size()[0]
        task_positions, task_orientations = poses.split([3, 4], dim=1)
        
        euler_angles_rpy = self.quaternion_to_euler_tensor(task_orientations)
        roll_angles = euler_angles_rpy[:, 0]
        pitch_angles = euler_angles_rpy[:, 1]
        yaw_angles = euler_angles_rpy[:, 2]

        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)
        task_rot_world = torch.zeros((batch_size, 2, 2), device=self.device)
        task_rot_world[:, 0, 0] = cos_yaw
        task_rot_world[:, 0, 1] = -sin_yaw
        task_rot_world[:, 1, 0] = sin_yaw
        task_rot_world[:, 1, 1] = cos_yaw

        print(f"{pitch_angles=}\n{roll_angles=}\n{task_positions[:, 2]=}")
        task_tensor = torch.stack([task_positions[:, 2]/max_height, pitch_angles/(torch.pi/2), roll_angles/torch.pi], dim=1)
        # task_tensor = self.task_embedding(task_tensor)

        return task_rot_world, task_tensor

    def quaternion_to_euler_tensor(self, quaternions) -> torch.Tensor:
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