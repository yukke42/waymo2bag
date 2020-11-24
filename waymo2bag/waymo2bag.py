import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import rosbag
import rospy
import sensor_msgs.point_cloud2 as pcl2
import tensorflow
import tf
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils

# There is no bounding box annotations in the No Label Zone (NLZ)
# if set True, points in the NLZ are filtered
FILTER_NO_LABEL_ZONE_POINTS = False

# The dataset contains data from five lidars
# one mid-range lidar (top) and four short-range lidars (front, side left, side right, and rear)
SELECTED_LIDAR_SENSOR = [
    dataset_pb2.LaserName.TOP,
    # dataset_pb2.LaserName.FRONT,
    # dataset_pb2.LaserName.SIDE_LEFT,
    # dataset_pb2.LaserName.SIDE_RIGHT,
    # dataset_pb2.LaserName.REAR,
]

# The value in the waymo open dataset is the raw intensity
NORMALIZE_INTENSITY = True

# Note: disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Waymo2Bag(object):

    def __init__(self, load_dir, save_dir, num_workers):
        # turn on eager execution for older tensorflow versions
        if int(tensorflow.__version__.split('.')[0]) < 2:
            tensorflow.enable_eager_execution()

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.num_workers = num_workers

        # tfrecord_file = 'tfrecord/segment-17612470202990834368_2800_000_2820_000_with_camera_labels.tfrecord'
        # tfrecord_file = 'tfrecord/segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord'
        # tfrecord_file = 'tfrecord/segment-12940710315541930162_2660_000_2680_000_with_camera_labels.tfrecord'
        # self.tfrecord_pathnames = [tfrecord_file]
        self.tfrecord_pathnames = glob.glob('tfrecord/*.tfrecord')

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def convert(self):
        print("start converting ...")
        # with Pool(self.num_workers) as p:
        #     list(tqdm.tqdm(p.imap(self.convert_tfrecord2bag, range(len(self))), total=len(self)))
        for i in range(len(self)):
            self.convert_tfrecord2bag(i)
        print("\nfinished ...")

    def convert_tfrecord2bag(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tensorflow.data.TFRecordDataset(pathname, compression_type='')
        len_dataset = len(list(dataset))

        filename = os.path.basename(pathname).split('.')[0]
        bag = rosbag.Bag('rosbag/' + str(filename) + '.bag', 'w', compression=rosbag.Compression.NONE)

        try:
            for frame_idx, data in enumerate(dataset):
                print('{} {}/{}'.format(file_idx, frame_idx, len_dataset))

                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                timestamp = rospy.Time.from_sec(frame.timestamp_micros * 1e-6)

                self.write_tf(bag, frame, timestamp)
                self.write_point_cloud(bag, frame, timestamp)

        finally:
            print(bag)
            bag.close()

    def write_tf(self, bag, frame, timestamp):
        """
        Args:
            bag (rosbag.Bag): bag to write
            frame (waymo_open_dataset.dataset_pb2.Frame): frame info
            timestamp (rospy.rostime.Time): timestamp of a frame
        """

        def get_static_transform(from_frame_id, to_frame_id, stamp, trans_mat):
            t = tf.transformations.translation_from_matrix(trans_mat)
            q = tf.transformations.quaternion_from_matrix(trans_mat)
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = from_frame_id
            tf_msg.child_frame_id = to_frame_id
            tf_msg.transform.translation.x = t[0]
            tf_msg.transform.translation.y = t[1]
            tf_msg.transform.translation.z = t[2]
            tf_msg.transform.rotation.x = q[0]
            tf_msg.transform.rotation.y = q[1]
            tf_msg.transform.rotation.z = q[2]
            tf_msg.transform.rotation.w = q[3]
            return tf_msg

        Tr_vehicle2world = np.array(frame.pose.transform).reshape(4, 4)

        transforms = [
            ('/world', '/base_link', np.linalg.inv(Tr_vehicle2world)),
            ('/base_link', '/velodyne', np.eye(4)),
        ]

        tf_message = TFMessage()
        for transform in transforms:
            _tf_msg = get_static_transform(
                from_frame_id=transform[0],
                to_frame_id=transform[1],
                stamp=timestamp,
                trans_mat=transform[2]
            )
            tf_message.transforms.append(_tf_msg)

        bag.write('/tf', tf_message, t=timestamp)

    def write_point_cloud(self, bag, frame, timestamp):
        """ parse and save the lidar data in psd format
        Args:
            bag (rosbag.Bag):
            frame (waymo_open_dataset.dataset_pb2.Frame):
            timestamp (rospy.rostime.Time): timestamp of a frame
        """

        range_images, camera_projections, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)
        ret_dict = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_indexes=(0, 1)
        )
        points = np.concatenate(ret_dict['points0'] + ret_dict['points1'], axis=0)
        intensity = np.concatenate(ret_dict['intensity0'] + ret_dict['intensity1'], axis=0)

        if NORMALIZE_INTENSITY:
            intensity = np.tanh(intensity)

        # concatenate x,y,z and intensity
        point_cloud = np.column_stack((points, intensity))

        header = Header()
        header.frame_id = 'velodyne'
        header.stamp = timestamp

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        pcl_msg = pcl2.create_cloud(header, fields, point_cloud)

        bag.write('/points_raw', pcl_msg, t=pcl_msg.header.stamp)


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_indexes=(0, 1)):
    """Convert range images to point cloud.
    modified from https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/range_image_utils.py#L612
    Args:
      frame: open dataset frame
       range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
       camera_projections: A dict of {laser_name,
         [camera_projection_from_first_return,
         camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_indexes: 0 for the first return, 1 for the second return.
    Returns:
      points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    tf = tensorflow
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    ret_dict = defaultdict(list)

    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2]
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation
    )

    for c in calibrations:
        if c.name not in SELECTED_LIDAR_SENSOR:
            continue
        for ri_index in ri_indexes:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            # No Label Zone
            if FILTER_NO_LABEL_ZONE_POINTS:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local
            )

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))

            ret_dict['points{}'.format(ri_index)].append(points_tensor.numpy())

            # Note: channel 1: intensity
            # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto#L176
            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1], tf.where(range_image_mask))
            ret_dict['intensity{}'.format(ri_index)].append(intensity_tensor.numpy())

    return ret_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('--save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = Waymo2Bag(args.load_dir, args.save_dir, args.num_workers)
    converter.convert()
