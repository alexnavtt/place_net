import open3d
import trimesh
import argparse
import numpy as np

def load_arguments():
    """
    Load the path to the config file from runtime arguments and load the config as a dictionary
    """
    parser = argparse.ArgumentParser(
        prog="estimate_pointcloud_normals.py",
        description="Script to estimate the normals for a generic pointcloud",
    )
    parser.add_argument('--pointcloud_file', help='Path to pointcloud file to load. Accepts all filetypes compatible with open3d.io.read_point_cloud')
    parser.add_argument('--pitch', default=0.02, type=float, help='voxel size to use for marching cubes')
    parser.add_argument('--output-path', help='path to save the generated pointcloud with normals')
    return parser.parse_args()

def main():
    args = load_arguments()

    if args.pointcloud_file is None:
        raise RuntimeError("No pointcloud file provided. Aborting")

    pc = open3d.io.read_point_cloud(args.pointcloud_file)
    if pc.has_normals():
        print("Pointcloud already has normals, exiting")
        exit(0)

    # Convert the pointcloud to a mesh using marching cubes
    trimesh_mesh = trimesh.voxel.ops.points_to_marching_cubes(np.asarray(pc.points), pitch=args.pitch)

    # Pointcloud
    pc_with_normals = open3d.geometry.PointCloud()
    pc_with_normals.points.extend(trimesh_mesh.vertices)
    pc_with_normals.normals.extend(trimesh_mesh.vertex_normals)

    open3d.visualization.draw([pc_with_normals])

    open3d.io.write_point_cloud(args.output_path, pc_with_normals, write_ascii=True)