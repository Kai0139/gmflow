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

        self.bridge = CvBridge()

        self.cv_img = None
        self.cv_thermal = None

        self.flow_sub = self.create_subscription(Float32MultiArray, "/img_with_flow", self.flow_callback, 1)
        self.img_sub = self.create_subscription(Image, "/camera/image_raw", self.img_callback, 1)
        self.thermal_img_sub = self.create_subscription(Image, "/optris/thermal_image", self.thermal_callback, 1)

        self.img_saved = False
        self.joint_img_timer = self.create_timer(0.01, self.timer_cb)
        

    def flow_callback(self, msg: Float32MultiArray):
        rec_time = time.time()
        if self.last_time is None:
            self.last_time = rec_time
            return
        print("t diff: {}".format(rec_time - self.last_time))
        self.last_time = rec_time

    def img_callback(self, msg):
        if self.cv_img is None:
            self.cv_img = self.bridge.imgmsg_to_cv2(msg)
            print("image captured")

    def thermal_callback(self, msg):
        if self.cv_thermal is None:
            self.cv_thermal = self.bridge.imgmsg_to_cv2(msg)
            self.cv_thermal = cv2.normalize(self.cv_thermal, None, 0, 255, cv2.NORM_MINMAX)
            self.cv_thermal = self.cv_thermal.astype(np.uint8)
            print("thermal captured")
            
    def timer_cb(self):
        if self.img_saved:
            return
        if self.cv_img is not None and self.cv_thermal is not None:
            self.cv_thermal = cv2.cvtColor(self.cv_thermal, cv2.COLOR_GRAY2RGB)
            print("visible image shape: {}".format(self.cv_img.shape))
            print("thermal image shape: {}".format(self.cv_thermal.shape))
            concat_img = np.concatenate((self.cv_img, self.cv_thermal), axis=1)
            print("concat image shape: {}".format(concat_img.shape))
            cv2.imwrite("visible_thermal.png", concat_img)
            self.img_saved = True
            
    def run(self):
        lr = self.create_rate(200)
        while rclpy.ok():
            print("lr")
            lr.sleep()
            # rclpy.spin_once(self)

    

if __name__ == "__main__":
    rclpy.init()
    fs = FlowSub()
    rclpy.spin(fs)
    fs.destroy_node()
    rclpy.shutdown()