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

    # The batch size to use for training
    batch_size: int = 1

    # The method to use for encoding the incoming pointcloud data
    encoder_type: Type[Union[PointNetEncoder, CNNEncoder]] = PointNetEncoder

    # Device to run on for PyTorch operations. 
    # NOTE: cuRobo (and hence ground truth calculations) require a CUDA device 
    device: torch.device = torch.device("cuda:0")

    # Learning rate to use during training
    learning_rate: float = 0.001

    # Whether to print debug statements and show visualizations
    debug: bool = False

    @staticmethod               
    def from_yaml_dict(yaml_config: dict):
        # Determine which type of pointcloud encoder to use
        pointcloud_encoder_lable: str = yaml_config['model_settings']['pointcloud_encoder']
        if pointcloud_encoder_lable.lower() == 'pointnet':
            pointcloud_encoder_type = PointNetEncoder
        elif pointcloud_encoder_lable.lower() in ['cnn', 'cnnencoder']:
            pointcloud_encoder_type = CNNEncoder

        try:
            # If the device is an integer, interpret it as a cuda device index
            torch_device = torch.device(f"cuda:{(int(yaml_config['model_settings']['cuda_device']))}")
        except ValueError:
            # Otherwise use the string as-is
            torch_device = torch.device(yaml_config['model_settings']['cuda_device'])

        task_geometry = yaml_config['task_geometry']
        return BaseNetModelConfig(
            robot_reach_radius=task_geometry['robot_reach_radius'],
            workspace_radius=task_geometry['workspace_radius'],
            workspace_height=task_geometry['base_link_elevation'] + task_geometry['robot_reach_radius'],
            workspace_floor=task_geometry['min_pointcloud_elevation'],
            batch_size=yaml_config['model_settings']['batch_size'],
            debug=yaml_config['model_settings']['debug'],
            learning_rate=yaml_config['model_settings']['learning_rate'],
            encoder_type=pointcloud_encoder_type,
            device=torch_device
        )
    
@dataclass
class TaskGenerationConfig:
    @dataclass
    class TaskGenerationCounts:
        # Poses sampled at a very small offset from points
        # in the pointcloud, and oriented with the end-effector
        # facing the surface. Roll is sampled randomly
        surface: int

        # Poses sampled at a small offset from points in the
        # pointcloud with a completely random orientation
        close: int

        # Poses sampled at a large offset from points in the
        # pointcloud with a completely random orientation. Note
        # that this distance should be less than the obstacle 
        # inclusion radius as there are other poses dedicated
        # to sampling empty space
        far: int

        # Poses sampled from other existing poses to have the
        # same x, y, and heading but with different elevation,
        # pitch and roll. This means that they will have 
        # identical obstacle encoding to other poses but different
        # pose encoding
        offset: int

    # Max and min distance of the end-effector frame from the nearest
    # point in the pointcloud for a sample to be considered valid
    @dataclass
    class TaskGenerationOffsets:
        # Distance from point on surface to sample pose 
        surface_offset: float

        # If there are no points within this radius, a sampled pose
        # is considered invalid 
        close_max     : float
        far_max       : float

        # If there is a point within this radius, a sampled pose is
        # considered invalid
        surface_min   : float
        close_min     : float
        far_min       : float

    counts: TaskGenerationCounts
    offsets: TaskGenerationOffsets
    max_ik_count: int | None = None

    @staticmethod
    def from_yaml_dict(yaml_config: dict):
        config = yaml_config['task_generation']

        if 'surface' in config['counts'] and 'surface_offset' not in config:
            raise RuntimeError(f'You specified that you wanted {config["counts"]["surface"]} surface poses, but you did not specify a surface offset')

        return TaskGenerationConfig(
            counts = TaskGenerationConfig.TaskGenerationCounts(
                surface= config['counts']['surface'] if 'surface' in config['counts'] else 0,
                close  = config['counts']['close'  ] if 'close'   in config['counts'] else 0,
                far    = config['counts']['far'    ] if 'far'     in config['counts'] else 0,
                offset = config['counts']['offset' ] if 'offset'  in config['counts'] else 0,
            ),

            offsets = TaskGenerationConfig.TaskGenerationOffsets(
                surface_min = config['min_offsets']['surface'] if 'surface' in config['min_offsets'] else 0.0,
                close_min   = config['min_offsets']['close'  ] if 'close'   in config['min_offsets'] else 0.0,
                far_min     = config['min_offsets']['far'    ] if 'far'     in config['min_offsets'] else 0.0,
                close_max   = config['max_offsets']['close'  ] if 'close'   in config['max_offsets'] else 1e10,
                far_max     = config['max_offsets']['far'    ] if 'far'     in config['max_offsets'] else 1e10,
                surface_offset = config['surface_offset'] if 'surface_offset' in config else 0.0
            ),

            max_ik_count = config['max_ik_count']
        )

@dataclass
class BaseNetConfig:
    # The list of all pointclouds associated with their filenames stored
    # in CPU tensors of shape (num_points, 6) arranged x, y, z, nx, ny, nz
    pointclouds: dict[str, open3d.geometry.PointCloud]

    # Modified cuRobot RobotConfig with the robot URDF and the robot
    # inverted robot URDF stored as a tuple in the kinematics debug field
    robot: RobotConfig

    # The configuration related specifically to the kinematics model, used
    # for both the PyTorch model as well as the cuRobo ground truth calculations
    model: BaseNetModelConfig

    # Configurations for sampling poses given an input pointcloud, including
    # distances to nearest obstacles, and proportions of poses at certain 
    # distances
    task_generation: TaskGenerationConfig

    # Location from which to load or to which to save tasks 
    task_path: str | None = None

    # Location from which to load or to which to save tasks 
    solution_path: str | None = None

    # All tasks associated with their pointclouds. Each task tensor
    # has the shape (num_tasks, 7) arranged as x, y, z, qw, qx, qy, qz
    tasks: dict[str, Tensor] | None = None

    # All task solutions associated with their task names. Each solution
    # tensor has the shape (num_tasks, num_pos, num_pos, num_headings) 
    # and represent the binary reachability value of that base pose in 
    # regular 3D grid of x, y, heading
    solutions: dict[str, Tensor] | None = None

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

    # Whether or not to include pointcloud information in the task generation
    # and PyTorch model. Set to False to learn just the robot kinematics
    check_environment_collisions: bool = True

    @staticmethod
    def from_yaml(filename: str, load_tasks = True, load_solutions = False):
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

        task_generation_config = TaskGenerationConfig.from_yaml_dict(yaml_config)

        return BaseNetConfig(
            pointclouds=pointclouds,
            task_path=yaml_config['task_data_path'],
            tasks=BaseNetConfig.load_tasks(yaml_config, pointclouds) if load_tasks else None,
            solution_path=yaml_config['solution_data_path'],
            solutions=BaseNetConfig.load_solutions(yaml_config, pointclouds) if load_solutions else None,
            robot=BaseNetConfig.load_robot_config(yaml_config, model_config.device),
            model=model_config,
            task_generation=task_generation_config,
            surface_task_offset=yaml_config['task_geometry']['surface_task_offset'],
            position_count=yaml_config['task_geometry']['position_count'],
            heading_count=yaml_config['task_geometry']['heading_count'],
            base_link_elevation=yaml_config['task_geometry']['base_link_elevation'],
            check_environment_collisions=yaml_config['task_geometry']['check_environment_collisions']
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
    def load_solutions(yaml_config: dict, pointclouds: dict):

        filepath = yaml_config['solution_data_path']
        solutions = {}

        for pointcloud_name in pointclouds.keys():
            solutions[pointcloud_name] = torch.load(os.path.join(filepath, f'{pointcloud_name}.pt'), map_location='cpu')

        return solutions
    
    @staticmethod
    def load_robot_config(yaml_config: dict, device: torch.device) -> RobotConfig:
        """
        Load the cuRobo config from the config yaml/XRDF file and the urdf specified in the config.
        This function inverts the loaded URDF such that the end effector becomes the base link 
        and the base link becomes the end effector. No modifications are needed from the user
        to either the URDF or the cuRobo config file.
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

        curobo_config = RobotConfig(
            kinematics=CudaRobotModelConfig.from_robot_yaml_file(
                file_path=robot_config,
                ee_link=base_link,
                urdf_path="/tmp/inverted_urdf.urdf",
                tensor_args=TensorDeviceType(device=device)
            )
        )

        # Make the URDF structure available later
        curobo_config.kinematics.kinematics_config.debug = (robot_urdf, inverted_robot_urdf)

        return curobo_config

        
