import open3d.visualization
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
        geometries.append(task_arrow.compute_triangle_normals())

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
        geometries.append(new_arrow.compute_triangle_normals())

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
            geometries.append(sphere_geom.compute_triangle_normals())

    return geometries

def urdf_pose_to_matrix(pose: urdfPose) -> np.ndarray:
        matrix = np.eye(4)
        if pose is not None:
            matrix[:3, :3] = scipy.spatial.transform.Rotation.from_euler(seq="zyx", angles=list(reversed(pose.rpy)), degrees=False).as_matrix()
            matrix[:3,  3] = np.array(pose.xyz)
        return matrix

def get_links_attached_to(link: str, robot_config: RobotConfig) -> dict[str, np.ndarray]:
    robot: Robot = robot_config.kinematics.kinematics_config.debug[0]

    def get_attached_child_links(link_name: str, transform_from_ee: dict) -> None:
        if link_name not in robot.child_map:
            return
        for child_joint, child_link in robot.child_map[link_name]:
            if robot.joint_map[child_joint].type == 'fixed' and child_link not in transform_from_ee:
                transform_from_ee[child_link] = transform_from_ee[link_name] @ urdf_pose_to_matrix(robot.joint_map[child_joint].origin)
                get_attached_child_links(child_link, transform_from_ee)

    def get_attached_parent_links(link_name: str, transform_from_ee: dict) -> None:
        if link_name not in robot.parent_map:
            return
        parent_joint, parent_link = robot.parent_map[link_name]
        if robot.joint_map[parent_joint].type == 'fixed':
            transform_from_ee[parent_link] = transform_from_ee[link_name] @ np.linalg.inv(urdf_pose_to_matrix(robot.joint_map[parent_joint].origin))
            get_attached_parent_links(parent_link, transform_from_ee)
            get_attached_child_links(parent_link, transform_from_ee)

    transform_from_ee = {link: np.eye(4)}
    get_attached_child_links(link, transform_from_ee)
    get_attached_parent_links(link, transform_from_ee) 

    return transform_from_ee

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
        robot_spheres = robot_model.get_robot_as_spheres(q=joint_state.cuda(robot_model.tensor_args.device))[0]
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

        link_pose = base_link_pose
        joint_idx = 0
        for link_idx, link_name in enumerate(chain_links):
            for attached_link, link_transform in get_links_attached_to(link_name, robot_config).items():
                for visual in robot_urdf.link_map[attached_link].visuals:
                    mesh = open3d.io.read_triangle_mesh(visual.geometry.filename[7:])
                    if visual.geometry.scale is not None:
                        mesh.scale(visual.geometry.scale)
                    mesh.transform(link_pose @ link_transform @ urdf_pose_to_matrix(visual.origin))
                    geometries.append(mesh)

            if link_idx != len(chain_joints):
                joint: Joint = robot_urdf.joint_map[chain_joints[link_idx]]
                link_pose = link_pose @ urdf_pose_to_matrix(joint.origin)

                if joint.type == 'fixed': continue

                joint_angle = joint_state[joint_idx].item()
                joint_idx += 1
                if joint.type == 'revolute':
                    link_pose[:3, :3] = link_pose[:3, :3] @ scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=np.array(joint.axis)*joint_angle, degrees=False).as_matrix()
                elif joint.type == 'prismatic':
                    link_pose[:3,  3] += link_pose[:3, :3] @ (joint_angle * np.array(joint.axis))

    return geometries

def visualize(*args):
    geometries = []

    for arg in args:
        # If the input is a list, we assume it is a list of already valid geometries
        if type(arg) == list:
            geometries += arg

        # If it's an open3d geometry we can just add it and move on
        elif issubclass(type(arg), open3d.geometry.Geometry):
            geometries.append(arg)

        # If the input is a tensor, it could be either pointcloud(s) or task_pose(s)
        elif type(arg) == torch.Tensor:
            last_dim = arg.size()[-1]

            # Pointcloud case
            if last_dim == 3 or last_dim == 6:
                has_normals = last_dim == 6

                # Determine if we have a batch or just one pointcloud
                if len(arg.size()) == 3:
                    pointclouds = arg
                else:
                    pointclouds = [arg]

                for pointcloud in pointclouds:
                    points = pointcloud[:, :3].cpu().numpy()
                    normals = pointcloud[:, 3:].cpu().numpy() if has_normals else None

                    new_pointcloud = open3d.geometry.PointCloud()
                    new_pointcloud.points.extend(points)
                    if normals is not None:
                        new_pointcloud.normals.extend(normals)
                    
                    geometries.append(new_pointcloud)

            # Task pose case
            if last_dim == 7:
                
                # Determine if we have a batch or just one pose
                if len(arg.size()) == 2:
                    poses = arg
                else:
                    poses = arg.unsqueeze(0)

                print(poses)
                geometries = geometries + get_task_arrows(cuRoboPose(position=poses[:, :3].cuda(), quaternion=poses[:, 3:].cuda()))

    open3d.visualization.draw(geometries)
