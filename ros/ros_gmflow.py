import numpy as np
from pathlib import Path
import cv2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

import torch
import torch.nn.functional as F

import sys
from pathlib import Path
gmflow_path = Path(__file__).resolve().parent.parent
print(gmflow_path)
sys.path.append(str(gmflow_path))

from gmflow.gmflow import GMFlow
from utils.flow_viz import flow_tensor_to_image

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

class GMFlowROS(object):
    def __init__(self, model_path: str) -> None:
        self.device = torch.device("cuda")
        self.feature_channels = 96
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
        self.model.load_state_dict(torch.load(model_path))
        
        self.attn_splits_list = [2]
        self.corr_radius_list = [-1]
        self.prop_radius_list = [-1]
        
        self.model.eval()
        self.bridge = CvBridge()

        self.img1 = None
        self.img2 = None
        self.save_idx = 0

        self.img_sub = rospy.Subscriber("/optris/thermal_image", Image, self.img_callback, queue_size=1)
        self.flow_pub = rospy.Publisher("/flow/compressed", CompressedImage, queue_size=1)

    def img_callback(self, img_msg):
        # self.save_image(img_msg)
        # return
        if self.img1 is None:
            self.img1 = self.load_img(img_msg)
            self.padder = InputPadder(self.img1.shape)
            print(self.img1.shape)
            self.img1 = self.padder.pad(self.img1)[0]
            return
        
        with torch.no_grad():
            self.img2 = self.padder.pad(self.load_img(img_msg))[0]
            results_dict = self.model(self.img1, self.img2,
                                 attn_splits_list=self.attn_splits_list,
                                 corr_radius_list=self.corr_radius_list,
                                 prop_radius_list=self.prop_radius_list,
                                 )
            
            flow_pred = results_dict["flow_preds"][-1]
            self.img1 = self.img2
            flow = self.padder.unpad(flow_pred[0]).cpu()
            self.visualize(flow)

    def load_img(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg)
        # cv_img = cv2.resize(cv_img, (int(cv_img.shape[1]/2), int(cv_img.shape[0]/2)))
        np_img = np.array(cv_img).astype(np.uint8)
        img = torch.from_numpy(np_img).permute(2, 0, 1).float()
        print("img size: {}".format(np_img.shape))
        return img[None].to(self.device)
    
    def visualize(self, flow):
        # flow = flow[0].permute(1,2,0).cpu().numpy()
        # print("flow shape: {}".format(flow.shape))
        # map flow to rgb image
        flow = flow_tensor_to_image(flow)
        # img_msg = self.bridge.cv2_to_imgmsg(flow, "bgr8")
        img_msg = self.bridge.cv2_to_compressed_imgmsg(flow)
        self.flow_pub.publish(img_msg)

if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent.parent.joinpath("train_results", "shrink_model", "step_200000.pth")
    # print("model path: ")
    rospy.init_node("gmflow_ros")
    gmf = GMFlowROS(str(model_path))
    rospy.spin()
