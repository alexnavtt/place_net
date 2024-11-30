import os
from glob import glob
from setuptools import setup
from generate_parameter_library_py.setup_helper import generate_parameter_module

generate_parameter_module(
    "base_net_ros_parameters",
    "base_net_ros/base_net_params.yaml"
)

package_name = 'base_net_ros'

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.py')),
    ],
    install_requires=['setuptools', 'base_net', 'rclpy'],
    zip_safe=True,
    author='Alex Navarro',
    author_email='alexnavtt@utexas.edu',
    maintainer='Alex Navarro',
    maintainer_email='alexnavtt@utexas.edu',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: BSD 3-Clause License',
        'Programming Language :: Python',
    ],
    description='A ROS2 interface for the base_net package',
    license='BSD',
    entry_points={
        'console_scripts': [
            'base_net_server = base_net_ros.base_net_server:main',
        ],
    },
)