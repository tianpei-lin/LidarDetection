# The functions only work in python2!

import fastbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2


def getOdomFromFastbag(bag_path, odom_topic):
    bag = fastbag.Reader(bag_path)
    bag.open()

    bag.read_messages(topics=[odom_topic])

    odom_list = []
    for topic, msg, _ in bag.read_messages(topics=[odom_topic]):
        timestamp = msg.header.stamp.to_sec()
        pos = [msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               msg.pose.pose.position.z]
        quat = [msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w]
        odom_list.append((timestamp, (pos, quat)))

    return odom_list


def getLidarPointsFromFastbag(bag_path, lidar_topics):
    bag = fastbag.Reader(bag_path)
    bag.open()

    lidar_list = []
    for topic, msg, t in bag.read_messages(topics=lidar_topics):
        timestamp = msg.header.stamp.to_sec()
        lidar = pc2.read_points(msg, skip_nans=True,
                                field_names=("x", "y", "z", "intensity"))
        lidar_points = [(p[0], p[1], p[2], p[3]) for p in lidar]
        lidar_list.append((topic, (timestamp, lidar_points), t.to_sec()))

    return lidar_list
