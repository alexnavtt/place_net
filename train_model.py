import torch
from models.pointcloud_encoder import PointNetEncoder, CNNEncoder
from models.base_net import BaseNet

def main():
    point = torch.Tensor([1, 2, 3])
    normal = torch.rand(3)
    normal /= normal.norm()
    orientation = torch.rand(4)
    orientation /= orientation.norm()

    point1 = torch.cat([torch.Tensor([4, 5, 6]), normal])
    point2 = torch.cat([torch.Tensor([7, 8, 9]), normal])

    pointcloud = torch.cat([point1, point2]).reshape([1, 6, 2])
    print(pointcloud, pointcloud.size())

    base_net_model = BaseNet(
        workspace_origin=torch.Tensor([0, 0, 0]),
        workspace_dim=torch.Tensor([1.9, 1.9, 1.9]),
        encoder_type=PointNetEncoder
    )

    base_net_model.forward(pointcloud, torch.cat([point, orientation]))

if __name__ == "__main__":
    main()