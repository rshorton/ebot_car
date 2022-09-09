from setuptools import setup
from glob import glob

package_name = 'ebot_car'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shorton',
    maintainer_email='horton.rscotti@gmail.com',
    description='Elsabot Donkey Car Control Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ebot_car = ebot_car.donkey_car:main',
        ],
    },
    scripts=['ebot_car/datastore.py',
             'ebot_car/datastore_v2.py',
             'ebot_car/tub_v2.py']
)
