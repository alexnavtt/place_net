import torch
from models.pointcloud_encoder import PointNetEncoder, CNNEncoder

class BaseNet(torch.nn.Module):
    def __init__(self, workspace_origin: torch.Tensor, workspace_dim: torch.Tensor, encoder_type: PointNetEncoder | CNNEncoder = PointNetEncoder):
        super(BaseNet, self).__init__()
        self.pointcloud_encoder = encoder_type()
        self.workspace_origin = workspace_origin
        self.workspace_dim = workspace_dim

    def forward(self, points: torch.Tensor, task: torch.Tensor):
        # Encode the points into a feature vector
        x = self.pointcloud_encoder(points)

        # Normalize the geometry information
        point, orientation = task.split([3, 4], dim=0)
        point = (point - self.workspace_origin) / self.workspace_dim
        orientation = -1*orientation if orientation[0] < 0 else orientation

        # Concatenate it all together
        x = torch.cat([x, point, orientation], dim=0)
