import os
from dataclasses import dataclass
from typing_extensions import Type, Union

import yaml
import torch
import open3d
import numpy as np
from torch import Tensor
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from base_net.models.pointcloud_encoder import PointNetEncoder, CNNEncoder
from base_net.utils.invert_robot_model import main as invert_urdf

# Allow running even without ROS
try:
    from ament_index_python import get_package_share_directory
except ModuleNotFoundError:
    pass

@dataclass
class BaseNetModelConfig:
    # The radial dimension of the robot's reachable space
    robot_reach_radius: float

    # The radial dimension of the workspace to consider from the task pose
    workspace_radius: float

    # The minimum height to consider for points in the input pointclouds
    workspace_floor: float

    # The maximum height to consider for points in the input pointclouds
    workspace_height: float

    # The method to use for encoding the incoming pointcloud data
    encoder_type: Type[Union[PointNetEncoder, CNNEncoder]] = PointNetEncoder

    # Device to run on for PyTorch operations. 
    # NOTE: cuRobo (and hence ground truth calculations) require a CUDA device 
    device: str = "cuda:0"

    # Whether to print debug statements and show visualizations
    debug: bool = False

    @staticmethod               
    def from_yaml_dict(yaml_config: dict):
        # Determine which type of pointcloud encoder to use
        pointcloud_encoder_lable: str = yaml_config['model_settings']['model_settings']
        if pointcloud_encoder_lable.lower() == 'pointnet':
            pointcloud_encoder_type = PointNetEncoder
        elif pointcloud_encoder_lable.lower() in ['cnn', 'cnnencoder']:
            pointcloud_encoder_type = CNNEncoder

        try:
            # If the device is an integer, interpret it as a cuda device index
            torch_device = f"cuda:{int(yaml_config['model_settings']['cuda_device'])}"
        except ValueError:
            # Otherwise use the string as-is
            torch_device = yaml_config['model_settings']['cuda_device']

        task_geometry = yaml_config['task_geometry']
        return BaseNetModelConfig(
            robot_reach_radius=task_geometry['robot_reach_radius'],
            workspace_radius=task_geometry['workspace_radius'],
            workspace_height=task_geometry['base_link_elevation'] + task_geometry['robot_reach_radius'],
            workspace_floor=task_geometry['min_pointcloud_elevation'],
            encoder_type=pointcloud_encoder_type,
            device=torch_device
        )

@dataclass
class BaseNetConfig:
    # The list of all pointclouds associated with their filenames stored
    # in CPU tensors of shape (num_points, 6) arranged x, y, z, nx, ny, nz
    pointclouds: dict[str, open3d.geometry.PointCloud]

    # All tasks associated with their pointclouds. Each task tensor
    # has the shape (num_tasks, 7) arranged as x, y, z, qw, qx, qy, qz
    tasks: dict[str, Tensor]

    # Modified cuRobot RobotConfig with the robot URDF and the robot
    # inverted robot URDF stored as a tuple in the kinematics debug field
    robot: RobotConfig

    # The configuration related specifically to the kinematics model, used
    # for both the PyTorch model as well as the cuRobo ground truth calculations
    model: BaseNetModelConfig

    # Used in task pose generation. The distance offset from surfaces to
    # place a task pose for the subset of tasks sampled as 'surface tasks'
    surface_task_offset: float = 0.10

    # Used in ground truth calculations. The number of position cells
    # in the x and y directions when calculating inverse kinematics. This
    # parameter is ignored by the PyTorch BaseNet model
    position_count: int = 20

    # Used in ground truth calculations. The number of discretized cells
    # in the heading direction when calculating inverse kinematics. This
    # parameter is ignored by the PyTorch BaseNet model
    heading_count: int = 20

    # Used in ground truth calculations. The elevation above the floor
    # plane at which the base link of the kinematic chain sits
    base_link_elevation: float = 0.0

    @staticmethod
    def from_yaml(filename: str):
        with open(filename, 'r') as f:
            yaml_config = yaml.safe_load(f)

        model_config = BaseNetModelConfig.from_yaml_dict(yaml_config)

        pointclouds = {}
        for pointcloud_config in yaml_config['pointclouds']:
            name, pointcloud = BaseNetConfig.load_pointcloud(
                filepath=pointcloud_config['path'], 
                min_elevation=model_config.workspace_floor, 
                max_elevation=model_config.workspace_height, 
                elevation_offset=pointcloud_config['elevation']
            )

            if name in pointclouds:
                raise RuntimeError(f"Duplicate pointcloud name detected: {name}")
            pointclouds[name] = pointcloud

        return BaseNetConfig(
            pointclouds=pointclouds,
            tasks=BaseNetConfig.load_tasks(yaml_config, pointclouds),
            robot=BaseNetConfig.load_robot_config(yaml_config),
            model=model_config,
            surface_task_offset=yaml_config['task_geometry']['surface_task_offset'],
            position_count=yaml_config['task_geometry']['position_count'],
            heading_count=yaml_config['task_geometry']['heading_count'],
            base_link_elevation=yaml_config['task_geometry']['base_link_elevation']
        )

    @staticmethod
    def load_pointcloud(filepath: str, min_elevation: float, max_elevation: float, elevation_offset: float) -> tuple[str, open3d.geometry.PointCloud]:
        height_filter = open3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-1e10, -1e10, min_elevation], 
            max_bound=[1e10, 1e10, max_elevation]
        )

        _, filename = os.path.split(filepath)
        name, extension = os.path.splitext(filename)

        if not os.path.isfile(filepath):
            raise RuntimeError(f"Cannot load pointcloud from file \"{filepath}\" - it does not exist")
        if extension != '.pcd':
            raise RuntimeError(f"Cannot load pointcloud from file \"{filepath}\" - unsupported extension \"{extension}\"")

        pointcloud_o3d = open3d.io.read_point_cloud(filepath)
        if not pointcloud_o3d.has_normals():
            # pointcloud_o3d.estimate_normals()
            raise RuntimeError(f"Cannot load pointcloud from file \"{filepath}\" - it does not contain normals")
        
        pointcloud_o3d.translate([0, 0, -elevation_offset])
        pointcloud_o3d = pointcloud_o3d.crop(height_filter)
        pointcloud_o3d = pointcloud_o3d.voxel_down_sample(0.05)
        return name, pointcloud_o3d
    
    @staticmethod 
    def load_tasks(yaml_config: dict, pointclouds: dict):

        filepath = yaml_config['task_data_path']
        tasks = {}

        for pointcloud_name in pointclouds.keys():
            with open(os.path.join(filepath, f'{pointcloud_name}.task'), 'r') as f: 
                task_config = yaml.safe_load(f)

            num_tasks = len(task_config)
            task_tensor = torch.empty([num_tasks, 7])

            for idx in range(num_tasks):
                task_tensor[idx, :3] = torch.Tensor(task_config[idx]['position'])
                task_tensor[idx, 3:] = torch.Tensor(task_config[idx]['orientation'])

            tasks[pointcloud_name] = task_tensor

        return tasks
    
    @staticmethod
    def load_robot_config(yaml_config: dict) -> RobotConfig:
        """
        Load the cuRobo config from the config yaml/XRDF file and the urdf specified in the config.
        This function inverts the loaded URDF such that the end effector becomes the base link 
        and the base link becomes the end effector. No modifications are needed from the user
        to either the URDF or the cuRobot config file.
        """
        
        # Resolve ros package paths if necessary
        if 'ros_package' in yaml_config['curobo_config_file']:
            curobo_file = os.path.join(
                get_package_share_directory(yaml_config['curobo_config_file']['ros_package']),
                yaml_config['curobo_config_file']['path']
            )
        else:
            curobo_file = yaml_config['curobo_config_file']['path']

        urdf_config = yaml_config['urdf_file']
        if 'ros_package' in urdf_config:
            urdf_file = os.path.join(
                get_package_share_directory(urdf_config['ros_package']),
                urdf_config['path']
            )
        else:
            urdf_file = urdf_config['path']
        
        # Load and process the cuRobo config file
        with open(curobo_file) as f:
            robot_config = yaml.safe_load(f)

        curobo_config_extension = os.path.splitext(curobo_file)[1]
        if curobo_config_extension == '.xrdf':
            ee_link = robot_config['tool_frames'][0]
            for config_item in robot_config['modifiers']:
                if 'set_base_frame' in config_item:
                    base_link = config_item['set_base_frame']
                    config_item['set_base_frame'] = ee_link
                    robot_config['tool_frames'][0] = base_link
                    break

            # Temporary workaround because cuRobo doesn't properly process an XRDF dict
            with open('/tmp/robot_xrdf.xrdf', 'w') as f:
                yaml.dump(robot_config, f)
            robot_config = '/tmp/robot_xrdf.xrdf'

        elif curobo_config_extension == '.yaml':
            ee_link = robot_config['robot_cfg']['kinematics']['ee_link']
            base_link = robot_config['robot_cfg']['kinematics']['base_link']
            robot_config['robot_cfg']['kinematics']['ee_link'] = base_link
            robot_config['robot_cfg']['kinematics']['base_link'] = ee_link
        else:
            raise RuntimeError(f'Received cuRobo config file with unsupported extension: "{curobo_config_extension}"')

        # Load and process the URDF file
        robot_urdf, inverted_robot_urdf = invert_urdf(
            urdf_path    = urdf_file, 
            xacro_args   = urdf_config['xacro_args'] if 'xacro_args' in urdf_config else '', 
            end_effector = ee_link, 
            output_path  = "/tmp/inverted_urdf.urdf"
        )

        curobo_config = RobotConfig(kinematics=CudaRobotModelConfig.from_robot_yaml_file(
            file_path=robot_config,
            ee_link=base_link,
            urdf_path="/tmp/inverted_urdf.urdf")
        )

        # Make the URDF structure available later
        curobo_config.kinematics.kinematics_config.debug = (robot_urdf, inverted_robot_urdf)

        return curobo_config

        
