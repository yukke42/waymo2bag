# waymo2bag

Convert [WAYMO Open  Dataset](https://waymo.com/open/) dataset to ROS bag file the easy way!



## Requirements

- Docker (tested on  19.03.14)



## Setup

Download from here or create `waymo_open_dataset` package for Python2, because packages for Python3 are only provided officially. 



if you create a package:

```
cd build_wod_pkg
./build_pkg.sh
# waymo_open_dataset packages are saved to /tmp/pip_pkg_build
```



setup a package:

```
mkdir -p pip_pkg_build
mv /path/to/waymo_open_dataset_tf_2_0_0-1.2.0-cp27-cp27mu-manylinux2010_x86_64.whl pip_pkg_build/waymo_open_dataset_tf_2_0_0-1.2.0-cp27-cp27mu-manylinux2010_x86_64.whl
```



build docker for waymo2bag:

```
docker build -f docker/Dockerfile -t waymo2bag .
```



## How to use

You can convert tfrecord to rosbag with a following command:

All tfrecord files in `/path/to/tfrecord` are converted to rosbag.

```
docker run \
  -v /path/to/tfrecord:/data/tfrecord \
  -v ${PWD}/rosbag:/data/rosbag \
  -it waymo2bag waymo2bag --load_dir /data/tfrecord --save_dir /data/rosbag
```



if you want to run docker interactively:

```
docker run \
  -v /path/to/tfrecord:/data/tfrecord \
  -v ${PWD}/rosbag:/data/rosbag \
  -it waymo2bag bash
```



## Reference

 * [tomas789/kitti2bag](https://github.com/tomas789/kitti2bag)
 * [caizhongang/waymo_kitti_converter](https://github.com/caizhongang/waymo_kitti_converter)

