import argparse
import json
import math
import copy
from pathlib import Path

import numpy as np
import torch
import cv2
import rosbag
import sensor_msgs.point_cloud2 as pc2

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.utils.data_viz import plot_gt_boxes, plot_gt_det_cmp
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils.tracker_for_inference import TrackingManager


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--bag_file', type=str, default=None, help='specify the bag file to be inferenced')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for inference')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='model checkpoint')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()

    log_file = 'log_bag_inference.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    # Build dataset config
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=args.workers, training=False
    )

    # Build network
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # Initialize tracking manager
    tracking_manager = TrackingManager(cfg)

    # Load the bag data
    bag_data = rosbag.Bag(args.bag_file, 'r')

    json_dict = {'objects': []}
    frame_idx = 0
    object_id = 0
    odom_tmp = []

    # Save video
    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = cv2.VideoWriter('inf_result.avi', fourcc, 10.0, (231, 1601))

    # Inference with model
    with torch.no_grad():
        # load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        # start evaluation
        class_names = cfg.CLASS_NAMES
        point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        processor = DataProcessor(cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range, training=False)
        print("Predicting from bag file ...")

        for topic, msg, _ in bag_data.read_messages(topics=["/unified/lidar_points", "/navsat/odom"]):
            if topic == "/navsat/odom":
                odom_tmp = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]

            if topic == "/unified/lidar_points":
                timestr = '%0.9f' % msg.header.stamp.to_sec()
                timestr = timestr.split('.')

                lidar_pts_unified = pc2.read_points(msg)
                lidar_pts_unified = np.array(list(lidar_pts_unified))[:, :4]
                lidar_intensity = copy.deepcopy(lidar_pts_unified[:, 3])
                lidar_pts_unified[:, 3] = 1
                lidar_pts_unified = np.matmul(lidar_pts_unified, tf_matrix_world2imu.T)
                lidar_pts_unified[:, 3] = 0

                # lidar_pts_unified[:, 2] += 0.3
                # lidar_pts_unified = np.concatenate((lidar_pts_unified, np.zeros((lidar_pts_unified.shape[0], 1))), axis=1)
                batch_dict = processor.forward({'points': lidar_pts_unified, 'use_lead_xyz': True, 'batch_size': 1})
                batch_dict['points'] = np.concatenate(
                    (np.zeros((batch_dict['points'].shape[0], 1)), batch_dict['points']), axis=1)
                batch_dict['voxel_coords'] = np.concatenate(
                    (np.zeros((batch_dict['voxel_coords'].shape[0], 1)), batch_dict['voxel_coords']), axis=1)
                load_data_to_gpu(batch_dict)

                print("Predicting message %04d" % frame_idx)
                pred_dicts, _ = model(batch_dict)

                # Update the tracking manager
                tracked_objects = tracking_manager.update_tracking(pred_dicts)

                # Save video
                if SAVE_VIDEO:
                    points = batch_dict['points'][:, 1:4].cpu().detach().numpy()
                    det_boxes = tracked_objects['pred_boxes']
                    # det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
                    bev_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
                    frame = plot_gt_boxes(points, det_boxes, bev_range, ret=True)
                    video_output.write(frame)

                ###########################Plot DET results###############################
                PLOT_BOX = False
                if PLOT_BOX:
                    points = batch_dict['points'][:, 1:4].cpu().detach().numpy()
                    # det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
                    det_boxes = tracked_objects['pred_boxes']
                    bev_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
                    plot_gt_boxes(points, det_boxes, bev_range, name="%04d" % frame_idx)
                ##########################################################################

                # Format the det result
                for obj_idx in range(tracked_objects['pred_boxes'].shape[0]):
                    # Traverse all objects in json_dict's object list, find the item with same uuid
                    FIND_IN_ARCHIVE = False
                    for archived_object in json_dict['objects']:
                        if tracked_objects['object_ids'][obj_idx] == int(archived_object['uuid']):
                            bound_info = {'Tr_imu_to_world': {'qw': odom_tmp[0], 'qx': odom_tmp[1],
                                                              'qy': odom_tmp[2], 'qz': odom_tmp[3],
                                                              'x': odom_tmp[4], 'y': odom_tmp[5],
                                                              'z': odom_tmp[6]},
                                          'timestamp': int(timestr[0]),
                                          'timestamp_nano': int(timestr[1]),
                                          'velocity': {'x': 0, 'y': 0, 'z': 0}}
                            obj_loc = tracked_objects['pred_boxes'][obj_idx, :3].tolist()
                            obj_dim = tracked_objects['pred_boxes'][obj_idx, 3:6].tolist()
                            obj_rz = tracked_objects['pred_boxes'][obj_idx, 6].tolist()

                            # Rotate the object center
                            loc_x = obj_loc[0] * math.cos(-obj_rz) - obj_loc[1] * math.sin(-obj_rz)
                            loc_y = obj_loc[0] * math.sin(-obj_rz) + obj_loc[1] * math.cos(-obj_rz)

                            bound_info.update(
                                {'center': {'x': loc_x, 'y': loc_y, 'z': obj_loc[2]},
                                 'direction': {'x': 0, 'y': 0, 'z': 0},
                                 'heading': obj_rz,
                                 'is_front_car': 0,
                                 'position': {'x': obj_loc[0], 'y': obj_loc[1], 'z': obj_loc[2]},
                                 'size': {'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]},
                                 })

                            archived_object['bounds'].append(bound_info)
                            FIND_IN_ARCHIVE = True
                            break

                    if FIND_IN_ARCHIVE:
                        continue

                    # If not find, create new object info
                    new_object_info = {'bounds': [{'Tr_imu_to_world': {'qw': odom_tmp[0], 'qx': odom_tmp[1],
                                                                       'qy': odom_tmp[2], 'qz': odom_tmp[3],
                                                                       'x': odom_tmp[4], 'y': odom_tmp[5],
                                                                       'z': odom_tmp[6]},
                                                   'timestamp': int(timestr[0]),
                                                   'timestamp_nano': int(timestr[1]),
                                                   'velocity': {'x': 0, 'y': 0, 'z': 0}}
                                                  ],
                                       'size': {},
                                       'uuid': str(tracked_objects['object_ids'][obj_idx])
                                       }
                    obj_loc = tracked_objects['pred_boxes'][obj_idx, :3].tolist()
                    obj_dim = tracked_objects['pred_boxes'][obj_idx, 3:6].tolist()
                    obj_rz = tracked_objects['pred_boxes'][obj_idx, 6].tolist()

                    # Rotate the object center
                    loc_x = obj_loc[0] * math.cos(-obj_rz) - obj_loc[1] * math.sin(-obj_rz)
                    loc_y = obj_loc[0] * math.sin(-obj_rz) + obj_loc[1] * math.cos(-obj_rz)

                    new_object_info['bounds'][0].update(
                        {'center': {'x': loc_x, 'y': loc_y, 'z': obj_loc[2]},
                         'direction': {'x': 0, 'y': 0, 'z': 0},
                         'heading': obj_rz,
                         'is_front_car': 0,
                         'position': {'x': obj_loc[0], 'y': obj_loc[1], 'z': obj_loc[2]},
                         'size': {'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]},
                         })
                    new_object_info['size'].update({'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]})
                    json_dict['objects'].append(new_object_info)
                    object_id += 1
                frame_idx += 1
                odom_tmp = []

    if SAVE_VIDEO:
        video_output.release()
        print("Inference results saved as video.")

    json_txt = json.dumps(json_dict, indent=4)
    with open('%s.json' % args.bag_file, 'w') as f:
        f.write(json_txt)
        print("JSON file saved at %s.json" % args.bag_file)
    bag_data.close()


if __name__ == '__main__':
    tf_matrix_world2imu = np.array([9.7664633748321206e-01, 2.3700882187393947e-02,
                                    2.1354233630479907e-01, 4.4884194774399999e+00,
                                    -2.7655994825909448e-02, 9.9949637395898994e-01,
                                    1.5552890021958202e-02, -1.9965142422800002e-02,
                                    -2.1306615826035888e-01, -2.1095415271440141e-02,
                                    9.7680992196315319e-01, 2.8337476145100000e+00,
                                    0., 0., 0., 1.]).reshape([4, 4])
    main()
