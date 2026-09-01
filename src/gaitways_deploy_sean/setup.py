import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'gaitways_deploy_sean'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join(package_name, 'launch', '*launch.py'))),
    ],
    install_requires=['setuptools', 'crccheck', 'torch'],
    zip_safe=True,
    maintainer='sean',
    maintainer_email='luckysandon3@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'policy_deploy = gaitways_deploy_sean.policy_deploy:main'
        ],
    },
)
