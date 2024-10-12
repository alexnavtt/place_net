import rclpy
import open3d
import trimesh
import numpy as np
import rclpy.publisher
import rclpy.node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points, create_cloud, structured_to_unstructured

fields = [PointField(name='x' , offset=0, datatype=PointField.FLOAT32, count=1),
          PointField(name='y' , offset=4, datatype=PointField.FLOAT32, count=1),
          PointField(name='z' , offset=8, datatype=PointField.FLOAT32, count=1),
          PointField(name='normal_x', offset=12, datatype=PointField.FLOAT32, count=1),
          PointField(name='normal_y', offset=16, datatype=PointField.FLOAT32, count=1),
          PointField(name='normal_z', offset=20, datatype=PointField.FLOAT32, count=1)]

def pointcloud_callback(publisher: rclpy.publisher.Publisher, msg: PointCloud2):
    numpy_pointcloud = read_points(msg, field_names=['x', 'y', 'z'])
    numpy_pointcloud = structured_to_unstructured(numpy_pointcloud, dtype=float)

    open3d_pointcloud = open3d.geometry.PointCloud()
    open3d_pointcloud.points.extend(numpy_pointcloud)
    open3d_pointcloud.remove_non_finite_points()
    search = open3d.geometry.KDTreeSearchParamKNN(50)
    open3d_pointcloud.estimate_normals(search_param=search, fast_normal_computation=False)
    open3d_pointcloud.orient_normals_towards_camera_location()
    output_cloud = create_cloud(msg.header, fields, np.concatenate([open3d_pointcloud.points, open3d_pointcloud.normals], axis=1))
    
    publisher.publish(output_cloud)

def main():
    rclpy.init()
    node = rclpy.node.Node("pointcloud_republish_node")
    pub = node.create_publisher(PointCloud2, "/cloud_out", 10)
    sub = node.create_subscription(PointCloud2, "/cloud_in", lambda msg: pointcloud_callback(pub, msg), 10)

    node.get_logger().info("Started pointcloud republisher")
    rclpy.spin(node)

if __name__ == "__main__":
    main()