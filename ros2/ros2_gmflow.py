import numpy as np
from pathlib import Path
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, Float64MultiArray, MultiArrayDimension

import torch
import torch.nn.functional as F

import sys
from pathlib import Path
gmflow_path = Path(__file__).resolve().parent.parent
print(gmflow_path)
sys.path.append(str(gmflow_path))

from gmflow.gmflow import GMFlow
from utils.flow_viz import flow_tensor_to_image, flow_to_image

import time

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', padding_factor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class GMFlowROS(Node):
    def __init__(self, model_path: str) -> None:
        super().__init__("gmflow_node")
        self.device = torch.device("cuda")
        self.feature_channels = 128
        self.num_scales = 1
        self.upsample_factor = 8
        self.num_head = 1
        self.attention_type = "swin"
        self.ffn_dim_expansion = 4
        self.num_transformer_layers = 6
        self.model = GMFlow(feature_channels=self.feature_channels,
                   num_scales=self.num_scales,
                   upsample_factor=self.upsample_factor,
                   num_head=self.num_head,
                   attention_type=self.attention_type,
                   ffn_dim_expansion=self.ffn_dim_expansion,
                   num_transformer_layers=self.num_transformer_layers,
                   ).to(self.device)
        print("load model: {}".format(model_path))
        weights = torch.load(model_path)["model"]
        self.model.load_state_dict(weights)
        
        self.attn_splits_list = [2]
        self.corr_radius_list = [-1]
        self.prop_radius_list = [-1]
        
        self.model.eval()
        self.bridge = CvBridge()

        self.img1_ts = None
        self.img2_ts = None
        self.img1_arr_rgb = None
        self.img2_arr_rgb = None
        self.img1_arr_mono = None
        self.img2_arr_mono = None
        self.img1_tensor = None
        self.img2_tensor = None
        self.save_idx = 0

        self.ts_offset = 1643e6

        self.img_sub = self.create_subscription(Image, "/optris/thermal_image", self.img_callback, 1)
        self.flow_pub = self.create_publisher(Image, "/flow", 1)
        self.img_flow_pub = self.create_publisher(Float32MultiArray, "/img_with_flow", 1)

    def img_callback(self, img_msg):
        print(img_msg.header.stamp)
        # self.save_image(img_msg)
        # return
        if self.img1_tensor is None:
            self.img1_arr_rgb, self.img1_arr_mono = self.load_img_arr(img_msg)
            self.img1_ts = Time.from_msg(img_msg.header.stamp).nanoseconds / 1e9
            self.img1_tensor = self.load_img_tensor(self.img1_arr_rgb)
            self.padder = InputPadder(self.img1_tensor.shape)
            # print(self.img1.shape)
            self.img1_tensor = self.padder.pad(self.img1_tensor)[0]
            print("padded img shape: {}".format(self.img1_tensor.shape))
            return
        
        with torch.no_grad():
            self.img2_ts = Time.from_msg(img_msg.header.stamp).nanoseconds / 1e9
            self.img2_arr_rgb, self.img2_arr_mono = self.load_img_arr(img_msg)
            self.img2_tensor = self.padder.pad(self.load_img_tensor(self.img2_arr_rgb))[0]
            
            inf_start = time.time()
            results_dict = self.model(self.img1_tensor, self.img2_tensor,
                                 attn_splits_list=self.attn_splits_list,
                                 corr_radius_list=self.corr_radius_list,
                                 prop_radius_list=self.prop_radius_list,
                                 )
            t_cost = time.time() - inf_start
            print("inference time cost: {}".format(t_cost))

            flow_pred = results_dict["flow_preds"][-1]
            flow_pred = self.padder.unpad(flow_pred[0])
            print("flow_pred shape: {}".format(flow_pred.shape))
            flow_data = flow_pred.cpu().numpy()

            self.publish_multiarray(self.img1_arr_mono, self.img2_arr_mono, self.img1_ts, self.img2_ts, flow_data)
            t_cost = time.time() - inf_start
            print("send data time cost: {}".format(t_cost))

            self.img1_tensor = self.img2_tensor
            self.img1_arr_rgb = self.img2_arr_rgb
            self.img1_arr_mono = self.img2_arr_mono
            self.img1_ts = self.img2_ts
            
            # print("flow dims: {}".format(flow.shape))
            self.visualize(flow_pred)

    def load_img_arr(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg) / 64
        cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX)
        cv_img_255 = cv_img.astype(np.uint8)
        cv_img_rgb = cv2.cvtColor(cv_img_255, cv2.COLOR_GRAY2RGB)
        print("rgb cv_img shape: {}".format(cv_img_rgb.shape))
        # cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX)
        # print(img_msg.encoding)
        np_img_rgb = np.array(cv_img_rgb).astype(np.float32)
        return np_img_rgb, cv_img

    def load_img_tensor(self, np_img):
        img = torch.from_numpy(np_img)
        img = img.permute(2, 0, 1).float()
        # print("img size: {}".format(img.shape))
        return img[None].to(self.device)
    
    def visualize(self, flow):
        # flow = flow[0].permute(1,2,0).cpu().numpy()
        print("flow input shape: {}".format(flow.shape))
        # map flow to rgb image
        flow = flow_tensor_to_image(flow)
        print("flow image shape: {}".format(flow.shape))
        # print("converted flow dims: {}".format(flow.shape))
        img_msg = self.bridge.cv2_to_imgmsg(flow, "bgr8")
        # img_msg = Image()
        # img_msg.data = flow
        # img_msg = self.bridge.cv2_to_compressed_imgmsg(flow)
        self.flow_pub.publish(img_msg)

    def publish_flow_image(self, img1, img2, img1_ts, img2_ts, flow):
        flow_img = Image()

        pass

    def publish_multiarray(self, img1, img2, img1_ts, img2_ts, flow):
        print("img1 ts: {}".format(img1_ts))
        print("img2 ts: {}".format(img2_ts))
        multi_arr = Float32MultiArray()
        dim1 = MultiArrayDimension()
        dim1.label = "height"
        dim1.size = img1.shape[0]
        dim1.stride = img1.shape[0] * img1.shape[1] * 4
        dim2 = MultiArrayDimension()
        dim2.label = "width"
        dim2.size = img1.shape[1]
        dim2.stride = img1.shape[1] * 4
        dim3 = MultiArrayDimension()
        dim3.label = "channels"
        dim3.size = 4
        dim3.stride = 4
        multi_arr.layout.dim.append(dim1)
        multi_arr.layout.dim.append(dim2)
        multi_arr.layout.dim.append(dim3)

        img1_data = img1.reshape((img1.shape[0] * img1.shape[1]))
        img2_data = img2.reshape((img2.shape[0] * img2.shape[1]))
        flow_x_data = flow[0, :, :].reshape((img1.shape[0] * img1.shape[1]))
        flow_y_data = flow[1, :, :].reshape((img1.shape[0] * img1.shape[1]))
        # flow_data = flow.reshape((flow.shape[0] * flow.shape[1] * flow.shape[2]))
        multi_arr.data = img1_data.tolist() + img2_data.tolist() + \
            flow_x_data.tolist() + flow_y_data.tolist() + [img1_ts - self.ts_offset, img2_ts - self.ts_offset]
        self.img_flow_pub.publish(multi_arr)
        pass

if __name__ == "__main__":
    rclpy.init()
    model_path = Path(__file__).resolve().parent.parent.joinpath("pretrained", "gmflow_things-e9887eda.pth")
    # print("model path: ")
    gmf = GMFlowROS(str(model_path))
    rclpy.spin(gmf)

    gmf.destroy_node()
    rclpy.shutdown()