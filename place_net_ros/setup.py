import os
import pathlib
from glob import glob
from setuptools import setup
from generate_parameter_library_py.setup_helper import generate_parameter_module

generate_parameter_module(
    "place_net_ros_parameters",
    "place_net_ros/place_net_params.yaml"
)

package_name = 'place_net_ros'

# Record the source path so we can use it to launch docker containers
src_path = pathlib.Path(__file__).resolve().parent
docker_path = src_path.parent.joinpath('place_net', 'docker')
local_path = os.path.join(src_path, 'source_directory.txt')
with open(local_path, 'w') as f:
    f.write(str(docker_path))

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name), ['source_directory.txt'])
    ],
    install_requires=['setuptools', 'place_net', 'rclpy'],
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
    description='A ROS2 interface for the place_net package',
    license='BSD',
    entry_points={
        'console_scripts': [
            'place_net_server = place_net_ros.place_net_server:main',
            'docker_server = place_net_ros.run_docker:main'
        ],
    },
)