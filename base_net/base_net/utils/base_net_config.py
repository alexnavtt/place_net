import os
import copy
import hashlib
from dataclasses import dataclass, field
from typing_extensions import Type, Union

import yaml
import torch
import open3d
import numpy as np
import scipy.spatial.transform
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from urdf_parser_py.urdf import Robot
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from base_net.models.pointcloud_encoder import PointNetEncoder, CNNEncoder
from base_net.utils.invert_robot_model import main as invert_urdf
from base_net.utils.pointcloud_region import PointcloudRegion
from base_net.models.loss import FocalLoss, DiceLoss, TverskyLoss

# Allow running even without ROS
try:
    from ament_index_python import get_package_share_directory
except ModuleNotFoundError:
    pass

def tensor_hash(tensor: Tensor):
    tensor_bytes = tensor.cpu().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

@dataclass
class BaseNetRobotConfig:
    robot: RobotConfig
    inverted_robot: RobotConfig
    urdf: Robot
    inverted_urdf: Robot

@dataclass
class BaseNetModelConfig:

    # Whether or not to use pointcloud normals for training and inference
    use_normals: bool = True

    # The length of the features to calculate for the pointcloud and tasks
    feature_size: int = 1024

    # The number of channels to use in the first level of the deconvolution network.
    # Each subsequent layer will have half the channels as the last
    channel_count: int = 256

    # Dropout probability for 3D deconvolution
    convolution_dropout: float = 0.0

    # The batch size to use for training
    batch_size: int = 1

    # Max number of epochs to use for training
    num_epochs: int = 200

    # Max number of epochs to go without improvement before ending training
    # Set to zero to disable early stopping
    patience: int = 0

    # The method to use for encoding the incoming pointcloud data
    encoder_type: Type[Union[PointNetEncoder, CNNEncoder]] = PointNetEncoder

    # Loss function type
    loss_fn_type: Type[Union[BCEWithLogitsLoss, DiceLoss, FocalLoss]] = DiceLoss

    # Device to run on for PyTorch operations. 
    device: torch.device = torch.device("cuda:0")

    # Learning rate to use during training
    learning_rate: float = 0.001

    # Whether or not to use a separate classifier to determine if there are any reachable poses
    external_classifier: bool = False

    # The path to use as a base for saving tensorboard progress during training
    log_base_path: str | None = None

    # The path to use as a base for saving model checkpoints during training
    checkpoint_base_path: str | None = None

    # How many epochs between saving checkpoints. Set to zero or null to disable
    checkpoint_frequency: int | None = None

    # How the split the data into training, validation, and testing
    data_split: list = field(default_factory=list)

    @staticmethod               
    def from_yaml_dict(yaml_config: dict):
        model_settings: dict = yaml_config['model_settings']

        # Determine which type of pointcloud encoder to use
        pointcloud_encoder_lable: str = model_settings['pointcloud_encoder']
        if pointcloud_encoder_lable.lower() == 'pointnet':
            pointcloud_encoder_type = PointNetEncoder
        elif pointcloud_encoder_lable.lower() in ['cnn', 'cnnencoder']:
            pointcloud_encoder_type = CNNEncoder

        try:
            # If the device is an integer, interpret it as a cuda device index
            torch_device = torch.device(f"cuda:{(int(model_settings['cuda_device']))}")
        except ValueError:
            # Otherwise use the string as-is
            torch_device = torch.device(model_settings['cuda_device'])

        # Determine which loss function to use
        loss_fn_label: str = model_settings.get('loss_function', 'dice').lower()
        match loss_fn_label:
            case 'bce' | 'bce_loss' | 'bceloss' | 'binary_cross_entropy' | 'binary-cross-entropy':
                pos_weight = model_settings['bce'].get('pos_weight', 1.0) if 'bce' in model_settings else 1.0
                pos_weight_tensor = torch.tensor([pos_weight], device=torch_device)
                loss_fn_type = lambda pos_weight=pos_weight_tensor: BCEWithLogitsLoss(pos_weight=pos_weight)
            case 'dice' | 'dice_loss' | 'diceloss':
                loss_fn_type = DiceLoss
            case 'focal' | 'focal_loss' | 'focalloss':
                pos_weight = model_settings['focal_loss'].get('pos_weight', 1.0) if 'focal_loss' in model_settings else 1.0
                loss_fn_type = lambda pos_weight=pos_weight, device=torch_device: FocalLoss(pos_weight=pos_weight, device=device)
            case 'tversky' | 'tversky_loss' | 'tverskyloss':
                alpha = model_settings['tversky_loss'].get('alpha', 0.5) if 'tversky_loss' in model_settings else 0.5
                beta = model_settings['tversky_loss'].get('beta', 0.5) if 'tversky_loss' in model_settings else 0.5
                loss_fn_type = lambda alpha=alpha, beta=beta: TverskyLoss(alpha, beta)
            case _:
                raise ValueError(f'Unrecognized loss function type passed: {model_settings["loss_function"]}')
            
        # Determine the name of the model path based on the hyperparameters
        batch_size = model_settings.get('batch_size', 1)
        use_normals = model_settings.get('use_normals', True)
        learning_rate = model_settings.get('learning_rate', 0.001)
        feature_size = model_settings.get('feature_size', 1024)
        channel_count = model_settings.get('channel_count', 256)
        convolution_dropout = model_settings.get('convolution_dropout', 0.0)
        external_classifier = model_settings.get('external_classifier', False)
        
        model_name = loss_fn_label
        model_name += '_b' + str(batch_size)
        model_name += '_lr' + f'{learning_rate:.10f}'.rstrip('0').lstrip('0').lstrip('.')
        model_name += '_f' + str(feature_size)
        model_name += '_c' + str(channel_count)
        model_name += '_d' + str(int(100*convolution_dropout))
        if not use_normals: model_name += '_nn'
        if external_classifier: model_name += '_pos'

        log_base_path = model_settings.get('log_base_path', None)
        if log_base_path is not None:
            log_base_path = os.path.join(log_base_path, model_name)

        checkpoint_base_path = model_settings.get('checkpoint_base_path', None)
        if checkpoint_base_path is not None:
            checkpoint_base_path = os.path.join(checkpoint_base_path, model_name)

        return BaseNetModelConfig(
            use_normals=use_normals,
            feature_size=feature_size,
            channel_count=channel_count,
            convolution_dropout=convolution_dropout,
            batch_size=batch_size,
            num_epochs=model_settings.get('num_epochs', 200),
            patience=model_settings.get('patience', 0),
            learning_rate=learning_rate,
            external_classifier=external_classifier,
            encoder_type=pointcloud_encoder_type,
            loss_fn_type=loss_fn_type,
            device=torch_device,
            log_base_path=log_base_path,
            checkpoint_base_path=checkpoint_base_path,
            checkpoint_frequency=model_settings.get('checkpoint_frequency', None),
            data_split=model_settings.get('data_split', [60, 20, 20])
        )
    
@dataclass
class TaskGeometryConfig:
    # The elevation above the ground at which the manipulator base link lies
    base_link_elevation: float

    # The maximum reach of the robot in the plane of the manipulator base link
    max_radial_reach: float

    # The minimum elevation at which tasks can be queried
    min_task_elevation: float

    # The maximum elevation at which tasks can be queried
    max_task_elevation: float

    # The minimum elevation of points to include in the pointcloud. Useful for
    # disregarding low profile obstacles and the floor which might otherwise 
    # count as obstacles when the really shouldn't
    min_pointcloud_elevation: float = 0.0

    # The maximum elevation of points to include in the pointcloud. Set to None
    # to have this automatically deduced as the base_link_elevation + max_vertical_reach
    _max_pointcloud_elevation: float | None = None

    # The planar radial distance from the task both in which to keep points in the 
    # pointcloud. This can be larger than the max_radial_reach if the manipulator 
    # base link has other collision links attached which need to be accounted for. 
    # Set to None to have this automatically deduced as max_radial_reach
    _max_pointcloud_radius: float | None = None

    @property
    def max_pointcloud_elevation(self):
        if self._max_pointcloud_elevation is not None:
            return self._max_pointcloud_elevation 
        else:
            return self.base_link_elevation + self.max_vertical_reach
        
    @property
    def max_pointcloud_radius(self):
        if self._max_pointcloud_radius is not None:
            return self._max_pointcloud_radius
        else:
            return self.max_radial_reach

    @staticmethod
    def from_yaml_dict(yaml_dict: dict):
        return TaskGeometryConfig(
            base_link_elevation = yaml_dict['task_geometry']['base_link_elevation'],
            max_radial_reach = yaml_dict['task_geometry']['max_radial_reach'],
            max_task_elevation = yaml_dict['task_geometry']['max_task_elevation'],
            min_task_elevation = yaml_dict['task_geometry']['min_task_elevation'],
            min_pointcloud_elevation = yaml_dict['task_geometry'].get('min_pointcloud_elevation', 0.0),
            _max_pointcloud_elevation = yaml_dict['task_geometry'].get('max_pointcloud_elevation', None),
            _max_pointcloud_radius = yaml_dict['task_geometry'].get('max_pointcloud_radius', None),
        )
    
@dataclass
class TaskGenerationConfig:
    # Whether or not to visualize each set of poses after sampling
    visualize: bool = False

    # The offset from surfaces to place surface points
    surface_point_offset: float | None = None

    # The minimum clearance for a surface point to be considered valid
    surface_point_clearance: float = 0.0

    # The number of surface points to sample
    surface_point_count: int = 0

    # A list of ranges from the environment at which to sample random poses
    # Each entry has fields 'name', 'offset_bounds', 'count'
    offset_points: list[dict] | None = None

    # The regions in which to sample poses
    regions: dict[str, PointcloudRegion] = field(default_factory=dict)

    # Device to run on for PyTorch operations. Must be a CUDA device
    device: torch.device = torch.device("cuda:0")

    @staticmethod
    def from_yaml_dict(yaml_config: dict, pointclouds: dict[str, open3d.geometry.PointCloud]):
        config: dict = yaml_config['task_generation']
        
        if 'surface_points' in config:
            surface_point_offset = config['surface_points']['offset']
            surface_point_clearance = config['surface_points'].get('min_clearance', 0.0)
            surface_point_count = config['surface_points']['count']
        else:
            surface_point_offset = surface_point_clearance = surface_point_count = None

        pointcloud_regions = {pointcloud_name: PointcloudRegion(pointcloud) for pointcloud_name, pointcloud in pointclouds.items()}
        for pointcloud_name, region in pointcloud_regions.items():
            if pointcloud_name not in config['pointcloud_regions']: continue

            region_configs = config['pointcloud_regions'][pointcloud_name]
            for region_config in region_configs:
                if 'min_bound' in region_config and 'max_bound' in region_config:
                    center = 0.5*(np.array(region_config['min_bound']) + np.array(region_config['max_bound']))
                    extent = np.array(region_config['max_bound']) - np.array(region_config['min_bound'])
                else:
                    center = np.array(region_config['center'])
                    extent = np.array(region_config['extent'])
                region.add_region(center, extent, region_config.get('yaw', 0.0))

        return TaskGenerationConfig(
            visualize=config.get('visualize', False),
            surface_point_offset=surface_point_offset,
            surface_point_clearance=surface_point_clearance,
            surface_point_count=surface_point_count,
            offset_points=config.get('offset_points', None),
            regions=pointcloud_regions
        )
    
@dataclass
class InverseReachabilityMapConfig:
    # The number of entries in the task directions [z, pitch, roll]
    task_resolution: dict[str, int]

    # The number of entries in the solution directions [x, y, yaw]
    solution_resolution: dict[str, int]

    @staticmethod
    def from_yaml_dict(yaml_dict: dict):
        return InverseReachabilityMapConfig(
            task_resolution=yaml_dict['inverse_reachability_map']['task_resolution'],
            solution_resolution=yaml_dict['inverse_reachability_map']['solution_resolution']
        )

@dataclass
class BaseNetConfig:
    # Used for saving and loading config. A direct copy of the yaml
    # file used to create this config object
    yaml_source: dict 

    # The list of all pointclouds associated with their filenames stored
    # in CPU tensors of shape (num_points, 6) arranged x, y, z, nx, ny, nz
    pointclouds: dict[str, open3d.geometry.PointCloud]

    # The robot model
    robot_config: BaseNetRobotConfig

    # The configuration related the PyTorch model 
    model: BaseNetModelConfig

    # The configuration related to the workspace bounds and robot geometry
    task_geometry: TaskGeometryConfig

    # Configurations for sampling poses given an input pointcloud
    task_generation: TaskGenerationConfig

    # Configurations for calculating ground truth values 
    inverse_reachability: InverseReachabilityMapConfig

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

    # The maximum number of IK solutions to work on in parallel. This 
    # is useful for computers with limited VRAM. Set to None to allow
    # unlimited parallel problems
    max_ik_count: int | None  = None

    # Whether or not to show debug visualizations during ground truth 
    # calculations, training, and testing
    debug: bool = False

    @staticmethod
    def from_yaml_file(filename: str, load_pointclouds = True, load_tasks = True, load_solutions = False, device=None):
        with open(filename, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if device is not None:
            yaml_config['model_settings']['cuda_device'] = device
        return BaseNetConfig.from_yaml_dict(yaml_config, load_pointclouds, load_tasks, load_solutions)

    @staticmethod
    def from_yaml_dict(yaml_config: dict, load_pointclouds = True, load_tasks = True, load_solutions = False):
        model_config = BaseNetModelConfig.from_yaml_dict(yaml_config)
        task_geometry = TaskGeometryConfig.from_yaml_dict(yaml_config)

        pointclouds = {}
        if load_pointclouds:
            for pointcloud_name, pointcloud_config in yaml_config['pointclouds'].items():
                name, pointcloud = BaseNetConfig.load_pointcloud(
                    filepath=os.path.join(yaml_config['pointcloud_data_path'], f'{pointcloud_name}.pcd'), 
                    min_elevation=task_geometry.min_pointcloud_elevation, 
                    max_elevation=task_geometry.max_pointcloud_elevation, 
                    pointcloud_config=pointcloud_config
                )

                if name in pointclouds:
                    raise RuntimeError(f"Duplicate pointcloud name detected: {name}")
                pointclouds[name] = pointcloud

        task_generation_config = TaskGenerationConfig.from_yaml_dict(yaml_config, pointclouds)
        tasks     = BaseNetConfig.load_tasks(yaml_config, pointclouds) if load_tasks else None
        solutions = BaseNetConfig.load_solutions(yaml_config, pointclouds, yaml_config.get('fake_solutions', False), tasks) if load_solutions else None

        return BaseNetConfig(
            yaml_source=copy.deepcopy(yaml_config),
            pointclouds=pointclouds,
            robot_config=BaseNetConfig.load_robot_config(yaml_config, model_config.device),
            model=model_config,
            task_geometry=task_geometry,
            task_generation=task_generation_config,
            inverse_reachability=InverseReachabilityMapConfig.from_yaml_dict(yaml_config),
            task_path=yaml_config.get('task_data_path', None),
            solution_path=yaml_config.get('solution_data_path', None),
            tasks=tasks,
            solutions=solutions,
            max_ik_count=yaml_config.get('max_ik_count', None),
            debug=yaml_config.get('debug', False)
        )

    @staticmethod
    def load_pointcloud(filepath: str, min_elevation: float, max_elevation: float, pointcloud_config: dict) -> tuple[str, open3d.geometry.PointCloud]:
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
            raise RuntimeError(f"Cannot load pointcloud from file \"{filepath}\" - it does not contain normals")
        
        pointcloud_o3d.translate([0, 0, -pointcloud_config.get('elevation', 0.0)])
        pointcloud_o3d = pointcloud_o3d.crop(height_filter)
        pointcloud_o3d = pointcloud_o3d.voxel_down_sample(0.05)
        if pointcloud_config.get('filter_statistical_outliers', False):
            std_dev = pointcloud_config.get('filter_std_dev', 1.5)
            num_neighbors = pointcloud_config.get('filter_num_neighbors', 10)
            pointcloud_o3d, _ = pointcloud_o3d.remove_statistical_outlier(num_neighbors, std_dev, True)
        return name, pointcloud_o3d

    @staticmethod
    def get_empty_env_task_grid(filepath: str):
        # Of shape (N, 3) with each tuple arranged as (z, pitch, roll)
        data = torch.load(os.path.join(filepath, 'empty_task.pt'))

        positions = torch.zeros(data.size(0), 3)
        positions[:, 2] = data[:, 0]

        rpy_vec = torch.zeros(data.size(0), 3)
        rpy_vec[:, 1:] = data[:, 1:]
        quaternions = torch.tensor(
            np.array([scipy.spatial.transform.Rotation.from_euler("ZYX", rpy, degrees=False).as_quat(scalar_first=True) for rpy in rpy_vec])
        )

        return torch.concatenate([positions, quaternions], dim=1).float()
    
    @staticmethod 
    def load_tasks(yaml_config: dict, pointclouds: dict) -> dict[str, Tensor]:

        filepath = yaml_config['task_data_path']
        tasks = {}

        # Then check if collision checked tasks are enabled
        for pointcloud_name in pointclouds.keys():
            tasks[pointcloud_name] = torch.load(os.path.join(filepath, f'{pointcloud_name}_task.pt'), map_location='cpu').float()

        return tasks
    
    @staticmethod 
    def load_solutions(yaml_config: dict, pointclouds: dict, fake_solutions: bool = False, tasks: dict[str, Tensor] = None):

        solution_filepath = yaml_config['solution_data_path']
        solutions = {}

        num_x = yaml_config['inverse_reachability_map']['solution_resolution']['x']
        num_y = yaml_config['inverse_reachability_map']['solution_resolution']['y']
        num_yaw = yaml_config['inverse_reachability_map']['solution_resolution']['yaw']

        # Then check if collision checked tasks are enabled
        for pointcloud_name in pointclouds.keys():
            if fake_solutions:
                solutions[pointcloud_name] = torch.randint(0, 2, (tasks[pointcloud_name].size(0), num_x, num_y, num_yaw), dtype=bool)
            else:
                solution_struct = torch.load(os.path.join(solution_filepath, f'{pointcloud_name}.pt'), map_location='cpu')
                current_task_hash = tensor_hash(tasks[pointcloud_name])
                if current_task_hash != solution_struct['task_hash']:
                    raise RuntimeError(f'Your loaded tasks for the environment {pointcloud_name} do not match the version used to calculate the ground truth values!')
                solutions[pointcloud_name] = solution_struct['solution_tensor']

        return solutions
    
    @staticmethod
    def load_robot_config(yaml_config: dict, device: torch.device) -> tuple[RobotConfig, RobotConfig]:
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

        # Get a list of all joints not in the manipulator chain that we want to set to a default value
        unused_joint_defaults = yaml_config.get('unused_joint_defaults', {})
        
        # Load and process the cuRobo config file
        with open(curobo_file) as f:
            inverted_robot_config = yaml.safe_load(f)

        curobo_config_extension = os.path.splitext(curobo_file)[1]
        if curobo_config_extension == '.xrdf':
            # XRDF doesn't allow cspace to mismatch with the URDF so we remove any that we are going to fix in place
            for joint_name, _ in unused_joint_defaults.items():
                if joint_name in inverted_robot_config['cspace']['joint_names']:
                    idx = inverted_robot_config['cspace']['joint_names'].index(joint_name)
                    del(inverted_robot_config['cspace']['joint_names'][idx])
                    del(inverted_robot_config['cspace']['acceleration_limits'][idx])
                    del(inverted_robot_config['cspace']['jerk_limits'][idx])

            # Issues with the base frame of the forward model mean we have to take the URDF root frame.
            # Otherwise collision bodies above (parents or children of parents of) the new base frame 
            # in the robot tree will cause an error loading the robot and the program will crash
            forward_robot_config = copy.deepcopy(inverted_robot_config)
            for idx, config_item in enumerate(forward_robot_config['modifiers']):
                if 'set_base_frame' in config_item:
                    print(f'NOTE: The modifier setting the robot base frame to {config_item["set_base_frame"]} has been removed from the forward robot model. This model is for debugging and does not affect normal operations')
                    del(forward_robot_config['modifiers'][idx])
                    break

            ee_link = inverted_robot_config['tool_frames'][0]
            for config_item in inverted_robot_config['modifiers']:
                if 'set_base_frame' in config_item:
                    base_link = config_item['set_base_frame']
                    config_item['set_base_frame'] = ee_link
                    inverted_robot_config['tool_frames'][0] = base_link
                    break

            # Temporary workaround because cuRobo doesn't properly process an XRDF dict
            with open('/tmp/forward_robot_xrdf.xrdf', 'w') as f:
                yaml.dump(forward_robot_config, f)
                forward_robot_config = '/tmp/forward_robot_xrdf.xrdf'
            with open('/tmp/inverted_robot_xrdf.xrdf', 'w') as f:
                yaml.dump(inverted_robot_config, f)
                inverted_robot_config = '/tmp/inverted_robot_xrdf.xrdf'

        elif curobo_config_extension == '.yaml':
            forward_robot_config = copy.deepcopy(inverted_robot_config)
            ee_link = inverted_robot_config['robot_cfg']['kinematics']['ee_link']
            base_link = inverted_robot_config['robot_cfg']['kinematics']['base_link']
            inverted_robot_config['robot_cfg']['kinematics']['ee_link'] = base_link
            inverted_robot_config['robot_cfg']['kinematics']['base_link'] = ee_link
        else:
            raise RuntimeError(f'Received cuRobo config file with unsupported extension: "{curobo_config_extension}"')

        # Load and process the URDF file
        forward_robot_urdf, inverted_robot_urdf = invert_urdf(
            urdf_path        = urdf_file, 
            xacro_args       = urdf_config['xacro_args'] if 'xacro_args' in urdf_config else '', 
            end_effector     = ee_link, 
            defaulted_joints = unused_joint_defaults
        )

        # Write the models to files to be used in loading
        forward_file_path = '/tmp/forward_urdf.urdf'
        with open(forward_file_path, 'w') as f:
            f.write(forward_robot_urdf.to_xml_string())
        inverse_file_path = '/tmp/inverse_urdf.urdf'
        with open(inverse_file_path, 'w') as f:
            f.write(inverted_robot_urdf.to_xml_string())

        forward_config = RobotConfig(
           kinematics=CudaRobotModelConfig.from_robot_yaml_file(
                file_path=forward_robot_config,
                ee_link=ee_link,
                urdf_path=forward_file_path,
                tensor_args=TensorDeviceType(device=device)
            ) 
        )

        inverted_config = RobotConfig(
            kinematics=CudaRobotModelConfig.from_robot_yaml_file(
                file_path=inverted_robot_config,
                ee_link=base_link,
                urdf_path=inverse_file_path,
                tensor_args=TensorDeviceType(device=device)
            )
        )

        # Issues with cuRobo robot loading prevent the forward robot model from being loaded
        return BaseNetRobotConfig(
            robot=forward_config,
            inverted_robot=inverted_config,
            urdf=forward_robot_urdf,
            inverted_urdf=inverted_robot_urdf
        )

        
