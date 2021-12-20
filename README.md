# waymo2bag

Convert [Waymo Open Dataset](https://waymo.com/open/) dataset to ROS bag file the easy way!

If you want to get rosbag2 data, use <https://gitlab.com/ternaris/rosbags>.

## Requirements

- Docker (tested on 19.03.14)

## Setup

build docker for waymo2bag:

```bash
docker build -f docker/Dockerfile -t waymo2bag .
```

## How to use

You can convert tfrecord to rosbag with a following command:

All tfrecord files in `/path_to_tfrecord` are converted to rosbag.

```bash
docker run \
  -v /path_to_tfrecord/data/tfrecord \
  -v ${PWD}/rosbag:/data/rosbag \
  -it waymo2bag waymo2bag
```

If you want to run docker interactively:

```bash
docker run \
  -v /path_to_tfrecord:/data/tfrecord \
  -v ${PWD}/rosbag:/data/rosbag \
  -it waymo2bag bash
```

Converted rosbag info:

```bash
$ rosbag info segment-10203656353524179475_7625_000_7645_000_with_camera_labels.bag
path:        segment-10203656353524179475_7625_000_7645_000_with_camera_labels.bag
version:     2.0
duration:    19.7s
start:       Apr 03 2018 01:53:34.97 (1522688014.97)
end:         Apr 03 2018 01:53:54.67 (1522688034.67)
size:        6.7 GB
messages:    2376
compression: none [1386/1386 chunks]
types:       sensor_msgs/Image       [060021388200f6f0f447d0fcd9c64743]
             sensor_msgs/PointCloud2 [1158d486dd51d683ce2f1be655c3c181]
             tf2_msgs/TFMessage      [94810edda583a504dfda3829e70d7eec]
topics:      /camera/front/image              198 msgs    : sensor_msgs/Image
             /camera/front_left/image         198 msgs    : sensor_msgs/Image
             /camera/front_right/image        198 msgs    : sensor_msgs/Image
             /camera/side_left/image          198 msgs    : sensor_msgs/Image
             /camera/side_right/image         198 msgs    : sensor_msgs/Image
             /lidar/concatenated/pointcloud   198 msgs    : sensor_msgs/PointCloud2
             /lidar/front/pointcloud          198 msgs    : sensor_msgs/PointCloud2
             /lidar/rear/pointcloud           198 msgs    : sensor_msgs/PointCloud2
             /lidar/side_left/pointcloud      198 msgs    : sensor_msgs/PointCloud2
             /lidar/side_right/pointcloud     198 msgs    : sensor_msgs/PointCloud2
             /lidar/top/pointcloud            198 msgs    : sensor_msgs/PointCloud2
             /tf                              198 msgs    : tf2_msgs/TFMessage

```

## Reference

- [tomas789/kitti2bag](https://github.com/tomas789/kitti2bag)
- [caizhongang/waymo_kitti_converter](https://github.com/caizhongang/waymo_kitti_converter)
