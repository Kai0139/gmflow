
from rosbags.highlevel import AnyReader
from cv_bridge import CvBridge
import cv2

import numpy as np

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



class FlowFromBag(object):
    def __init__(self) -> None:
        self.bag_path = Path('/home/viplab/data/thermal_slam_bags/ros2/xyz1')
        self.bridge = CvBridge()
        self.reader = AnyReader([self.bag_path])
        self.thermal_img_topic = '/optris/thermal_image'
        
        # print(self.connections)

        self.img1_captured = False
        self.img2_captured = False

        self.itr_idx = 0
        self.img1_idx = 8
        self.img2_idx = 15

        self.cv_img1 = None
        self.cv_img2 = None

        self.model_path = Path(__file__).resolve().parent.parent.joinpath("train_results", "shrink_model", "step_200000.pth")

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
        print("load model: {}".format(self.model_path))
        weights = torch.load(self.model_path)["model"]
        self.model.load_state_dict(weights)
        
        self.attn_splits_list = [2]
        self.corr_radius_list = [-1]
        self.prop_radius_list = [-1]
        
        self.model.eval()

        self.padder = None


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

    def show_images(self):
        cv2.imshow('img1', self.cv_img1)
        cv2.imshow('img2', self.cv_img2)
        cv2.waitKey(0)

    def load_img_tensor(self, np_img):
        img = np.array(np_img).astype(np.float32)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 2)
        img = img.permute(2, 0, 1).float()
        # print("img size: {}".format(img.shape))
        return img[None].to(self.device)

    def generate_flow(self):
        img1_tensor = self.load_img_tensor(self.cv_img1)
        img2_tensor = self.load_img_tensor(self.cv_img2)

        results_dict = self.model(img1_tensor, img2_tensor,
                                 attn_splits_list=self.attn_splits_list,
                                 corr_radius_list=self.corr_radius_list,
                                 prop_radius_list=self.prop_radius_list,
                                 )
        padder = InputPadder(img1_tensor.shape)
        flow_pred = results_dict["flow_preds"][-1]
        flow_pred = padder.unpad(flow_pred[0])
        flow_pred = flow_pred.permute(1, 2, 0)
        flow_data = flow_pred.detach().cpu().numpy()
        print("flow shape: {}".format(flow_data.shape))
        fs = cv2.FileStorage("flow_data.yml", cv2.FILE_STORAGE_WRITE)
        fs.write("flow", flow_data)


if __name__ == '__main__':
    flow_from_bag = FlowFromBag()
    flow_from_bag.read_bag()
    # flow_from_bag.show_images()
    flow_from_bag.generate_flow()