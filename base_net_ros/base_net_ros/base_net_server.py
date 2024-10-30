import os
import torch
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from base_net.models.base_net import BaseNet
from base_net.utils.base_net_config import BaseNetConfig

from base_net_msgs.srv import QueryBaseLocation

class BaseNetServer(Node):
    def __init__(self):
        super(BaseNetServer).__init__('base_net_server')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Load the model from the checkpoint path
        checkpoint_path: str = self.declare_parameter('checkpoint_path').value
        device_param: str = self.declare_parameter('device').value

        base_path, _ = os.path.split(checkpoint_path)
        self.base_net_config = BaseNetConfig.from_yaml_file(os.path.join(base_path, 'config.yaml'), load_solutions=False, load_tasks=False, device=device_param)
        self.base_net_model = BaseNet(self.base_net_config)

        checkpoint_config = torch.load(checkpoint_path, map_location=self.base_net_config.model.device)
        self.base_net_model.load_state_dict(checkpoint_config['base_net_model'])

        # Start up the ROS service
        self.base_location_server = self.create_service(QueryBaseLocation, '~/query_base_location', self.base_location_callback)

    def base_location_callback(self):
        """TODO"""

def main():
    rclpy.init()
    base_net_server = BaseNetServer('base_net_server')
    rclpy.spin(base_net_server)

if __name__ == '__main__':
    main()