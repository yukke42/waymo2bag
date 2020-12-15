from setuptools import setup

setup(
    name='waymo2bag',
    version='1.0',
    description='Convert Waymo dataset to ROS bag file the easy way!',
    author='Yusuke Muramatsu',
    url='',
    entry_points={
        'console_scripts': ['waymo2bag=waymo2bag.waymo2bag:waymo2bag'],
    },
)
