import numpy as np
from pathlib import Path
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, Float64MultiArray, MultiArrayDimension

import time


class FlowSub(Node):
    def __init__(self) -> None:
        super().__init__("flow_sub_node")
        self.last_time = None
        self.img_sub = self.create_subscription(Float32MultiArray, "/img_with_flow", self.img_callback, 1)

    def img_callback(self, msg: Float32MultiArray):
        rec_time = time.time()
        if self.last_time is None:
            self.last_time = rec_time
            return
        print("t diff: {}".format(rec_time - self.last_time))
        self.last_time = rec_time

if __name__ == "__main__":
    rclpy.init()
    fs = FlowSub()
    rclpy.spin(fs)
    fs.destroy_node()
    rclpy.shutdown()