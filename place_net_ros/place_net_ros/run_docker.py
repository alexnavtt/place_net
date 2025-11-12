import os
import atexit
import subprocess
from ament_index_python import get_package_share_directory

def main():
    this_pkg_share = get_package_share_directory('place_net_ros')
    with open(os.path.join(this_pkg_share, 'source_directory.txt')) as f:
        docker_path = f.read()
    
    def shutdown_callback():
        subprocess.run(['docker', 'container', 'stop', 'place_net_server_process'])
    atexit.register(shutdown_callback)
    
    env = os.environ.copy()
    env['PWD'] = docker_path
    subprocess.run(['docker', 'compose', 'run', '--name', 'place_net_server_process', '--rm', 'ros_server'], cwd=docker_path, env=env)

if __name__ == '__main__':
    main()