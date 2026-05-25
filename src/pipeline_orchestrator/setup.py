from setuptools import find_packages, setup

package_name = 'pipeline_orchestrator'

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
    description='Pipeline orchestrator for the AI planner.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator = pipeline_orchestrator.orchestrator:main',
            'graspgen_probe = pipeline_orchestrator.graspgen_probe:main',
            'graspgen_service = pipeline_orchestrator.graspgen_service:main',
            'graspgen_service_caller = pipeline_orchestrator.graspgen_service_caller:main',
        ],
    },
)
