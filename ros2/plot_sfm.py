from rosbags.highlevel import AnyReader
from cv_bridge import CvBridge
import cv2

import numpy as np

import torch
import torch.nn.functional as F

import sys
from pathlib import Path

class FlowFromBag(object):
    def __init__(self) -> None:
        self.bag_path = Path('/home/viplab/data/thermal_slam_bags/ros2/xyz1')
        self.bridge = CvBridge()
        self.reader = AnyReader([self.bag_path])
        self.thermal_img_topic = '/optris/thermal_image'
        
    def read_bag(self):
        self.reader.open()
        connections = [x for x in self.reader.connections if x.topic == self.thermal_img_topic]
        for connection, timestamp, rawdata in self.reader.messages(connections=connections):
            msg = self.reader.deserialize(rawdata, connection.msgtype)
            if not self.img1_captured and self.itr_idx == self.img1_idx:
                self.img1_captured = True
                self.cv_img1 = self.bridge.imgmsg_to_cv2(msg)
                self.cv_img1 = cv2.normalize(self.cv_img1, None, 0, 255, cv2.NORM_MINMAX)
                self.cv_img1 = self.cv_img1.astype(np.uint8)
                
            if not self.img2_captured and self.itr_idx == self.img2_idx:
                self.img2_captured = True
                self.cv_img2 = self.bridge.imgmsg_to_cv2(msg)
                self.cv_img2 = cv2.normalize(self.cv_img2, None, 0, 255, cv2.NORM_MINMAX)
                self.cv_img2 = self.cv_img2.astype(np.uint8)
            self.itr_idx += 1
        self.reader.close()


if __name__ == '__main__':
    flow_from_bag = FlowFromBag()
    flow_from_bag.read_bag()
    # flow_from_bag.show_images()
    flow_from_bag.generate_flow()