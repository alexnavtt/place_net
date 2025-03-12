import torch
import argparse
import numpy as np

from place_net.utils.place_net_config import PlaceNetConfig
from place_net.utils.task_visualization import get_robot_geometry_at_joint_state, visualize

def load_arguments():
    """
    Load the path to the config file from runtime arguments and load the config as a dictionary
    """
    parser = argparse.ArgumentParser(
        prog="visualize_spheres.py",
        description="Script to visualize the sphere collision geometry of the cuRobo config",
    )
    parser.add_argument('--config-file', help='configuration yaml file for the robot and task definitions')
    return parser.parse_args()

def main():
    args = load_arguments()
    model_config = PlaceNetConfig.from_yaml_file(args.config_file)

    joint_positions = torch.zeros(model_config.robot_config.inverted_robot.kinematics.kinematics_config.n_dof)
    visualize(get_robot_geometry_at_joint_state(model_config.robot_config, joint_positions, np.eye(4), inverted=False))

if __name__ == '__main__':
    main()