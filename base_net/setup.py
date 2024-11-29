from setuptools import setup, find_packages

setup(
    name='base_net',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy<2.0.0',
        'open3d==0.18.0',
        'torch',
        'xacro',
        'urdf_parser_py',
        'click',
        'scipy',
        'tensorboard',
        'setproctitle',
        'pyassimp'
    ],
    entry_points={
        'console_scripts': [
            'base_net=base_net.cli:main'
        ],
    },
    package_data={
        
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: BSD 3-Clause License',
        'Operating System :: Ubuntu',
    ],
    python_requires='>=3.6',
)
