#!/bin/bash

docker run \
  -v ${PWD}:/data \
  -v ${PWD}/tfrecord:/data/tfrecord \
  -v ${PWD}/rosbag:/data/rosbag \
  -it waymo2bag waymo2bag --load_dir /data/tfrecord --save_dir /data/rosbag

#docker run \
#  -v ${PWD}:/-data \
#  -v ${PWD}/tfrecord:/data/tfrecord \
#  -v ${PWD}/rosbag:/data/rosbag \
#  -it waymo2bag bash
