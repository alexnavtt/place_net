import os
import subprocess
from ament_index_python import get_package_share_directory

def main():
    this_pkg_share = get_package_share_directory('place_net_ros')
    with open(os.path.join(this_pkg_share, 'source_directory.txt')) as f:
        docker_path = f.read()
    subprocess.run(['docker', 'compose', 'run', '--rm', 'ros_server'], cwd=docker_path)

if __name__ == '__main__':
    main()