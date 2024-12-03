import os
import argparse
from base_net.utils.base_net_config import BaseNetConfig
from base_net.utils.inverse_reachability_map import InverseReachabilityMap

def load_arguments():
    """
    Load the path to the config file from runtime arguments and load the config as a dictionary
    """
    parser = argparse.ArgumentParser(
        prog="generate_inverse_reachability_map.py",
        description="Script to calculate the ground truth reachability values for BaseNet in the absence of obstacles",
    )
    parser.add_argument('--config-file', default='../config/task_definitions.yaml', help='configuration yaml file for the robot and task definitions')
    return parser.parse_args()

def main():
    args = load_arguments()
    model_config = BaseNetConfig.from_yaml_file(args.config_file, load_solutions=False, load_tasks=False)

    irm_config = model_config.inverse_reachability
    inverse_reachability_map = InverseReachabilityMap(
        min_elevation=model_config.task_geometry.min_task_elevation,
        max_elevation=model_config.task_geometry.max_task_elevation,
        reach_radius=model_config.task_geometry.max_radial_reach,
        xyz_resolution=(irm_config.solution_resolution['x'], irm_config.solution_resolution['y'], irm_config.task_resolution['z']),
        roll_pitch_yaw_resolution=(irm_config.task_resolution['roll'], irm_config.task_resolution['pitch'], irm_config.solution_resolution['yaw']),
    )

    inverse_reachability_map.solve(model_config.inverted_robot, model_config.task_geometry.base_link_elevation, model_config.max_ik_count)
    inverse_reachability_map.save(os.path.join(model_config.solution_path, 'base_net_irm.pt'))

if __name__ == "__main__":
    main()