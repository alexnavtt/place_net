import torch
from torch import Tensor
from base_net.utils import geometry

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

    def encode(self, poses: Tensor, max_height: float) -> tuple[Tensor, Tensor, Tensor]:
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
        # Handle the case of a 1D tensor
        if isinstance(poses.size(), int) or len(poses.size()) == 1:
            poses = poses.unsqueeze(0)

        batch_size = poses.size()[0]
        _, task_orientations = poses.split([3, 4], dim=1)
        
        yaw_angles = geometry.extract_yaw_from_quaternions(task_orientations)

        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)
        task_rot_world = torch.zeros((batch_size, 2, 2), device=poses.device, requires_grad=False)
        task_rot_world[:, 0, 0] = cos_yaw
        task_rot_world[:, 0, 1] = -sin_yaw
        task_rot_world[:, 1, 0] = sin_yaw
        task_rot_world[:, 1, 1] = cos_yaw

        adjusted_pose = geometry.encode_tasks(poses)
        z, pitch, roll = adjusted_pose.split([1, 1, 1], dim=1)
        task_encoding = torch.cat([(z - max_height/2)/max_height, pitch/(torch.pi/2), torch.sin(roll), torch.cos(roll)], dim=1)
        return task_rot_world, adjusted_pose, task_encoding

    def forward(self, encoded_poses: Tensor) -> Tensor:
        return self.task_embedding(encoded_poses)
