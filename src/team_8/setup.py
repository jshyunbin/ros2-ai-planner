import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'team_8'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml') + glob('config/*.yml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joshua Hyunbin Lee',
    maintainer_email='jshyunbin@gmail.com',
    description='Pipeline orchestrator for the AI planner.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator = team_8.orchestrator:main',
            'pose_probe = team_8.pose_probe:main',
            'curobo_service = team_8.curobo_service:main',
            'debug_viz = team_8.debug_viz:main',
            'graspgen_probe = team_8.graspgen_probe:main',
            'home_config_tuner = team_8.home_config_tuner:main',
            'graspgen_service = team_8.graspgen_service:main',
            'graspgen_service_caller = team_8.graspgen_service_caller:main',
            'segmentation_service = team_8.segmentation_service:main',
        ],
    },
)
