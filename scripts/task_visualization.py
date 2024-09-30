import torch
import open3d
import numpy as np
import scipy.spatial
from urdf_parser_py.urdf import Robot, Pose as urdfPose, Joint
from curobo.types.math import Pose as cuRoboPose
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

def get_task_arrows(task_poses: cuRoboPose) -> list[open3d.geometry.TriangleMesh]:
    geometries = []
    
    for task_idx in range(task_poses.batch):
        task_arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.015,
            cone_radius=0.025,
            cylinder_height=0.1,
            cone_height=0.05,
        )

        rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
        rotation = scipy.spatial.transform.Rotation.from_quat(quat=task_poses.quaternion[task_idx, :].squeeze().cpu().numpy(), scalar_first=True)
        task_arrow.rotate(rotation_to_x, center=[0, 0, 0])
        task_arrow.rotate(rotation.as_matrix(), center=[0, 0, 0])
        task_arrow.translate(task_poses.position[task_idx, :].squeeze().cpu().numpy())
        task_arrow.paint_uniform_color([0, 0, 1])
        geometries.append(task_arrow)

    return geometries

def get_base_arrows(pose: cuRoboPose, success: torch.Tensor | None = None) -> list[open3d.geometry.TriangleMesh]:
    geometries = []

    if success is None:
        success = torch.zeros(pose.batch)

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
    for position, rotation, pose_success in zip(pose.position, pose.quaternion, success):
        new_arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.03,
            cone_height=0.015,
            cylinder_split=1,
            resolution=4
        )
        new_arrow.paint_uniform_color([float(1-pose_success.item()), pose_success.item(), 0])
        new_arrow.rotate(rotation_to_x, center=[0, 0, 0])
        new_arrow.rotate(scipy.spatial.transform.Rotation.from_quat(rotation.cpu().numpy(), scalar_first=True).as_matrix(), center=[0, 0, 0])
        new_arrow.translate(position.cpu().numpy())
        geometries.append(new_arrow)

    return geometries

def get_spheres(spheres: torch.Tensor, task_poses: cuRoboPose) -> list[open3d.geometry.TriangleMesh]:
    geometries = []
    
    for task_idx in range(task_poses.batch):
        translation = task_poses.position[task_idx, :].squeeze().cpu().numpy()
        rotation = task_poses.quaternion[task_idx, :].squeeze().cpu().numpy()
        transform = np.eye(4)
        transform[:3, :3] = scipy.spatial.transform.Rotation.from_quat(rotation, scalar_first=True).as_matrix()
        transform[:3,  3] = translation

        for sphere in spheres:
            x, y, z, radius = sphere
            sphere_loc = transform @ np.array([x, y, z, 1])
            sphere_geom = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere_geom.translate(sphere_loc[:3])
            sphere_geom.paint_uniform_color([0.5, 0.5, 1.0])
            geometries.append(sphere_geom)

    return geometries

def get_robot_at_joint_state(
        robot_config: RobotConfig, 
        joint_state: torch.Tensor, 
        base_link_pose: np.ndarray,
        *, 
        inverted: bool = False,
        as_spheres: bool = True
    ) -> list[open3d.geometry.TriangleMesh]:

    robot_model = CudaRobotModel(config=robot_config.kinematics)
    geometries = []
    
    if as_spheres:
        robot_spheres = robot_model.get_robot_as_spheres(q=joint_state)[0]
        robot_spheres_o3d = [open3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius) for sphere in robot_spheres]
        for robot_sphere_o3d, robot_sphere in zip(robot_spheres_o3d, robot_spheres):
            robot_sphere_o3d.translate(robot_sphere.position)
            robot_sphere_o3d.vertex_colors.extend(np.random.rand(len(robot_sphere_o3d.vertices), 1).repeat(3, axis=1))
            geometries.append(robot_sphere_o3d)

    else:
        urdf_idx = 1 if inverted else 0
        robot_urdf: Robot = robot_config.kinematics.kinematics_config.debug[urdf_idx]
        chain_links = robot_urdf.get_chain(robot_config.kinematics.kinematics_config.base_link, robot_config.kinematics.kinematics_config.ee_link, links=True, joints=False)
        chain_joints = robot_urdf.get_chain(robot_config.kinematics.kinematics_config.base_link, robot_config.kinematics.kinematics_config.ee_link, links=False, joints=True)
        print(chain_links)

        def get_children_if_fixed(child_tuple):
            print(f"{child_tuple=}")
            links = []
            joints = []
            child_joint_name, child_link_name = child_tuple
            if robot_urdf.joint_map[child_joint_name].type != 'fixed':
                return [], []
            else:
                links.append(child_link_name)
                joints.append(child_joint_name)
                if child_link_name not in robot_urdf.child_map:
                    return links, joints
                for child in robot_urdf.child_map[child_link_name]:
                    new_links, new_joints = get_children_if_fixed(child)
                    joints = joints + new_joints
                    links = links + new_links
            return links, joints

        for child in robot_urdf.child_map[chain_links[-1]]:
            child_links, child_joints = get_children_if_fixed(child)
            chain_links = chain_links + child_links
            chain_joints = chain_joints + child_joints

        def urdf_pose_to_matrix(pose: urdfPose) -> np.ndarray:
            matrix = np.eye(4)
            if pose is not None:
                matrix[:3, :3] = scipy.spatial.transform.Rotation.from_euler(seq="zyx", angles=list(reversed(pose.rpy)), degrees=False).as_matrix()
                matrix[:3,  3] = np.array(pose.xyz)
            return matrix

        link_pose = base_link_pose
        joint_idx = 0
        for link_name, joint_name in zip(chain_links, chain_joints):
            for visual in robot_urdf.link_map[link_name].visuals:
                mesh = open3d.io.read_triangle_mesh(visual.geometry.filename[7:])
                mesh.transform(link_pose)
                geometries.append(mesh)

            joint: Joint = robot_urdf.joint_map[joint_name]
            link_pose = link_pose @ urdf_pose_to_matrix(joint.origin)

            if joint.type == 'fixed': continue

            joint_angle = joint_state[joint_idx].item()
            joint_idx += 1
            if joint.type == 'revolute':
                link_pose[:3, :3] = link_pose[:3, :3] @ scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=np.array(joint.axis)*joint_angle, degrees=False).as_matrix()
            elif joint.type == 'prismatic':
                link_pose[:3,  3] += link_pose[:3, :3] @ (joint_angle * np.array(joint.axis))

    return geometries

