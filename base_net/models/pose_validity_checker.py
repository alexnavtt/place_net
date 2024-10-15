import torch
from torch import Tensor

class PoseValidityChecker(torch.nn.Module):
    def __init__(self):
        super(PoseValidityChecker, self).__init__()

        self.pose_embedder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.pointcloud_attention = torch.nn.MultiheadAttention(
            embed_dim=256,
            num_heads=1,
            dropout=0.05,
            batch_first=True
        )

        self.bilinear = torch.nn.Bilinear(256, 256, 1024)

        self.classification_seq = torch.nn.Sequential(
            torch.nn.Linear(1024, 32),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 1)
        )

    def forward(self, pointcloud_embeddings: Tensor, pose_encodings: Tensor):
        # Embed pose from x, y, z, R, P, Y to a higher dimensional representation
        pose_embeddings: Tensor = self.pose_embedder(pose_encodings)

        # Determine which parts of the pointcloud are important for this problem
        attended_pointcloud, _ = self.pointcloud_attention(
            query=pointcloud_embeddings.unsqueeze(1),
            key=pose_embeddings.unsqueeze(1),
            value=pose_embeddings.unsqueeze(1)
        )

        # Now use these attended points to inform the robot reachability
        hidden_state: Tensor = self.bilinear(pose_embeddings, attended_pointcloud)

        # Classification network
        model_output = self.classification_seq(hidden_state)
        return model_output
