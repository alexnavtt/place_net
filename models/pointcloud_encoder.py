import torch
from torch.nn.functional import relu

class PointNetEncoder(torch.nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.num_channels = 3 + 3 # xyz plus normals

        """ Note: We skip the geometry and feature transform steps
            because our problem is not invariant to transformations
        """

        self.conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        batch_size, point_dim, num_points = x.size()
        assert(point_dim==6, "Points must be structured as xyz, normal-xyz tuples")

        # x is size (batch_size, 6, num_points)
        print(x)
        x = relu(self.bn1(self.conv1(x)))
        print(x)
        # x is size (batch_size, 64, num_points)
        x = relu(self.bn1(self.conv2(x)))
        # x is size (batch_size, 64, num_points)
        x = relu(self.bn2(self.conv3(x)))
        # x is size (batch_size, 128, num_points)
        x = self.bn3(self.conv4(x))
        # x is size (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)
        # x is size (batch_size, 1024, 1)
        x = x.view(-1, 1024)

        return x
    
class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.num_channels = 3 + 3 # xyz plus normals

        self.conv1 = torch.nn.Conv3d(in_channels=self.num_channels, out_channels=2, kernel_size=5, stride=3)
        self.conv2 = torch.nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1)