import copy
import torch
from torch import Tensor

from .pointcloud_encoder import PointNetEncoder
from .pose_encoder import PoseEncoder
from base_net.utils.base_net_config import BaseNetConfig

class PoseValidityChecker(torch.nn.Module):
    def __init__(self, config: BaseNetConfig):
        super(PoseValidityChecker, self).__init__()

        self.device = config.model.device
        self.feature_size = 1024
        self.pose_encoder = PoseEncoder(feature_size=self.feature_size)
        self.pointcloud_encoder = PointNetEncoder(feature_size=self.feature_size, use_normals=False)
        self.task_geometry = copy.deepcopy(config.task_geometry)

        self.classification_seq = torch.nn.Sequential(
            torch.nn.Linear(2*self.feature_size, self.feature_size),
            torch.nn.BatchNorm1d(self.feature_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(self.feature_size, self.feature_size),
            torch.nn.BatchNorm1d(self.feature_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(self.feature_size, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 1)
        )

        self.to(self.device)

    def forward(self, pointclouds: list[Tensor] | Tensor, tasks: Tensor):
        with torch.no_grad():
            # Embed the task poses and get the transforms needed for the pointclouds
            tasks = tasks.to(self.device)
            task_rotation, _, task_encoding = self.pose_encoder.encode(tasks, self.task_geometry.min_task_elevation ,self.task_geometry.max_task_elevation)

            # Preprocess the pointclouds to filter out irrelevant points and adjust the frame to be aligned with the task pose
            pointclouds = [pointcloud.to(self.device) for pointcloud in pointclouds]
            pointcloud_tensor, padding_mask = self.pointcloud_encoder.preprocess_inputs(pointclouds, task_rotation, tasks[:, :3], self.task_geometry)

        task_embedding: Tensor = self.pose_encoder(task_encoding)
        pointcloud_embedding: Tensor = self.pointcloud_encoder(pointcloud_tensor, padding_mask)

        combined_vector = torch.concatenate([pointcloud_embedding, task_embedding], dim=-1)

        # Classification network
        model_output = self.classification_seq(combined_vector)
        return model_output
