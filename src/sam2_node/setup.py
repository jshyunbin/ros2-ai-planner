from setuptools import find_packages, setup

package_name = 'sam2_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joshua Hyunbin Lee',
    maintainer_email='jshyunbin@gmail.com',
    description='SAM2 segmentation node stub.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sam2_node = sam2_node.sam2_node:main',
        ],
    },
)
