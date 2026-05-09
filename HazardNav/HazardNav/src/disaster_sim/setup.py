from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'disaster_sim'

scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name), ['README.md']),
    ],
    scripts=[
        os.path.join(scripts_dir, f)
        for f in os.listdir(scripts_dir)
        if f.endswith('.py')
    ] if os.path.isdir(scripts_dir) else [],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='disaster_sim',
    maintainer_email='clibang2022@163.com',
    description='Disaster diffusion simulation and gradient-guided autonomous navigation.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hazard_source_node = disaster_sim.hazard_source_node:main',
            'gradient_explorer_node = disaster_sim.gradient_explorer_node:main',
            'hazard_gazebo_visual_node = disaster_sim.hazard_gazebo_visual_node:main',
            'hazard_control_panel_node = disaster_sim.hazard_control_panel_node:main',
            'trajectory_logger_node = disaster_sim.trajectory_logger_node:main',
        ],
    },
)
