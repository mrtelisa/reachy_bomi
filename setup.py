from setuptools import find_packages, setup
from glob import glob

package_name = 'reachy_bomi'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elisa Martinenghi',
    maintainer_email='s6504193@studenti.unige.it',
    description='ROS 2 package for BoMI finger input to Reachy velocity commands',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'server_socket = reachy_bomi.server_socket:main',
            'cmd_vel_publisher = reachy_bomi.cmd_vel_publisher:main',
        ],
    },
)
