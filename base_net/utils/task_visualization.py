import os
import torch
import open3d
import collada
import numpy as np
import scipy.spatial
import collada.triangleset
import open3d.visualization
from urdf_parser_py.urdf import Robot, Pose as urdfPose, Joint
from curobo.types.math import Pose as cuRoboPose
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from base_net.utils.base_net_config import BaseNetConfig
from base_net.utils.pointcloud_region import PointcloudRegion

def get_task_arrows(task_poses: cuRoboPose | torch.Tensor, suffix: str = '') -> list[open3d.geometry.TriangleMesh]:

    if isinstance(task_poses, cuRoboPose):
        task_poses = torch.concatenate([task_poses.position, task_poses.quaternion], dim=1)
    
    if len(task_poses.size()) == 1:
        task_poses = task_poses.unsqueeze(0)

    arrows = open3d.geometry.TriangleMesh()    
    for task_pose in task_poses:
        task_arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.015,
            cone_radius=0.025,
            cylinder_height=0.1,
            cone_height=0.05,
        )
        task_pos, task_ori = torch.split(task_pose, [3, 4])

        rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
        rotation = scipy.spatial.transform.Rotation.from_quat(quat=task_ori.cpu().numpy(), scalar_first=True)
        task_arrow.rotate(rotation_to_x, center=[0, 0, 0])
        task_arrow.rotate(rotation.as_matrix(), center=[0, 0, 0])
        task_arrow.translate(task_pos.cpu().numpy())
        task_arrow.paint_uniform_color([0, 0, 1])
        arrows += task_arrow.compute_triangle_normals()

    return [{'name': f'task_arrows{suffix}', 'geometry': arrows, 'group': 'task_arrows'}]

def get_base_arrows(pose: cuRoboPose, success: torch.Tensor | None = None, prefix: str = '') -> list[open3d.geometry.TriangleMesh]:
    if success is None:
        success = torch.zeros(pose.batch)

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
    composite_mesh = open3d.geometry.TriangleMesh()
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

        composite_mesh += new_arrow.compute_triangle_normals()

    return [{'geometry': composite_mesh, 'name': f'{prefix}base_arrows', 'group': 'base_arrows'}]

def get_spheres(spheres: torch.Tensor, task_poses: torch.Tensor, color: list = [0.5, 0.5, 1.0], label = None) -> list[open3d.geometry.TriangleMesh]:
    all_spheres = open3d.geometry.TriangleMesh()
    for task_idx in range(task_poses.size(0)):
        translation = task_poses[task_idx, :3].cpu().numpy()
        rotation = task_poses[task_idx, 3:].cpu().numpy()
        transform = np.eye(4)
        transform[:3, :3] = scipy.spatial.transform.Rotation.from_quat(rotation, scalar_first=True).as_matrix()
        transform[:3,  3] = translation

        for sphere in spheres:
            x, y, z, radius = sphere
            sphere_loc = transform @ np.array([x, y, z, 1])
            sphere_geom = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere_geom.translate(sphere_loc[:3])
            sphere_geom.paint_uniform_color(color)
            all_spheres += sphere_geom.compute_triangle_normals()

    if label is not None:
        all_spheres = {'name': label, 'geometry': all_spheres, 'group': 'spheres'}

    return [all_spheres]

def get_regions(regions: PointcloudRegion) -> list[open3d.geometry.TriangleMesh]:
    transparent_material = open3d.visualization.rendering.MaterialRecord()
    transparent_material.shader = 'defaultLitTransparency'
    transparent_material.base_color = [0.0, 1.0, 0.0, 0.3]

    meshes = []
    for idx, region in enumerate(regions._regions):
        extent = region.extent
        new_box = open3d.geometry.TriangleMesh.create_box(width = extent[0], height=extent[1], depth=extent[2])
        new_box.translate(-0.5*extent)
        new_box.rotate(region.R)
        new_box.translate(region.center)
        new_box.compute_triangle_normals()
        meshes.append({'name': f'region_{idx}', 'geometry': new_box, 'material': transparent_material, 'group': 'regions'})

    return meshes

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

def collada_to_open3d(collada_geom: collada.Collada):    
    o3d_mesh = open3d.geometry.TriangleMesh()
    vertices_offset = 0
    for geom in collada_geom.geometries:
        for prim in geom.primitives:
            if isinstance(prim, collada.triangleset.TriangleSet):
                o3d_mesh.vertices.extend(prim.vertex)
                o3d_mesh.triangles.extend(prim.indices[:, :, 0] + vertices_offset)
                vertices_offset += len(prim.vertex)

                if hasattr(prim, 'colors') and prim.colors is not None:
                    o3d_mesh.vertex_colors.extend(prim.colors)

    if not o3d_mesh.has_vertex_normals():
        o3d_mesh.compute_vertex_normals()

    if collada_geom.assetInfo.upaxis == 'Z_UP':
        rotation = scipy.spatial.transform.Rotation.from_euler('XYZ', [90, 90, 0], degrees=True).as_matrix()
        o3d_mesh.rotate(rotation)
            
    return o3d_mesh

def get_robot_geometry_at_joint_state(
        robot_config: RobotConfig, 
        joint_state: torch.Tensor, 
        base_link_pose: np.ndarray,
        *, 
        inverted: bool = False
    ) -> list[open3d.geometry.TriangleMesh]:

    robot_model = CudaRobotModel(config=robot_config.kinematics)
    geometries = []
    
    robot_spheres = robot_model.get_robot_as_spheres(q=joint_state.cuda(robot_model.tensor_args.device))[0]
    robot_spheres_o3d = [open3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius) for sphere in robot_spheres]
    for robot_sphere_o3d, robot_sphere in zip(robot_spheres_o3d, robot_spheres):
        robot_sphere_o3d.translate(robot_sphere.position)
        robot_sphere_o3d.paint_uniform_color(np.random.rand(1).repeat(3))
        geometries.append({'geometry': robot_sphere_o3d.compute_triangle_normals(), 'group': 'robot_spheres', 'name': robot_sphere.name})

    urdf_idx = 1 if inverted else 0
    robot_urdf: Robot = robot_config.kinematics.kinematics_config.debug[urdf_idx]
    chain_links = robot_urdf.get_chain(robot_config.kinematics.kinematics_config.base_link, robot_config.kinematics.kinematics_config.ee_link, links=True, joints=False)
    chain_joints = robot_urdf.get_chain(robot_config.kinematics.kinematics_config.base_link, robot_config.kinematics.kinematics_config.ee_link, links=False, joints=True)

    link_pose = base_link_pose
    joint_idx = 0
    for link_idx, link_name in enumerate(chain_links):
        for attached_link, link_transform in get_links_attached_to(link_name, robot_config).items():
            for visual in robot_urdf.link_map[attached_link].visuals:
                visual_filename = visual.geometry.filename[7:]
                if os.path.splitext(visual_filename)[1] == '.dae':
                    try:
                        obj = collada.Collada(filename=visual_filename)
                        mesh = collada_to_open3d(obj)
                    except collada.common.DaeMalformedError:
                        continue
                else:
                    mesh = open3d.io.read_triangle_mesh(visual_filename)

                if visual.geometry.scale is not None:
                    mesh.scale(visual.geometry.scale)
                mesh.transform(link_pose @ link_transform @ urdf_pose_to_matrix(visual.origin))
                geometries.append({'geometry': mesh, 'group': 'robot_mesh', 'name': link_name})

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

def get_pointcloud(pointcloud_tensor: torch.Tensor, task: torch.Tensor, model_config: BaseNetConfig) -> list[open3d.geometry.Geometry]:
    # Filter the pointcloud to all points in the appropriate radius around the task
    distances = (pointcloud_tensor[:, :2] - task[:2]).norm(dim=1)
    valid_points = pointcloud_tensor[distances < model_config.task_geometry.max_pointcloud_radius, :]

    pointcloud = open3d.geometry.PointCloud()
    pointcloud.points.extend(valid_points[:, :3].cpu().numpy())
    pointcloud.normals.extend(valid_points[:, 3:].cpu().numpy())

    return [{'name': 'environment', 'geometry': pointcloud}]

def visualize(*args):
    geometries = []

    for arg in args:
        # If the input is a list, we assume it is a list of already valid geometries
        if type(arg) == list:
            geometries += arg

        # If it's an open3d geometry we can just add it and move on
        elif issubclass(type(arg), open3d.geometry.Geometry):
            geometries.append(arg)

        elif type(arg) == cuRoboPose:
            geometries = geometries + get_task_arrows(arg)

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

                geometries = geometries + get_task_arrows(cuRoboPose(position=poses[:, :3].cuda(), quaternion=poses[:, 3:].cuda()))

    open3d.visualization.draw(geometries)
