from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from cv_bridge import CvBridge
import cv2

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def closest(lst, K):
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K))

class FlowFromBag(object):
    def __init__(self) -> None:
        self.bag_name = "xyz_clip_400"
        self.name_idx_dict = {"xyz_clip_000": 17, "xyz_clip_100": 33, "xyz_clip_150": 32, "xyz_clip_200": 32, "xyz_clip_250": 32, "xyz_clip_300": 32, "xyz_clip_350": 32, "xyz_clip_400": 33}
        self.bag_path = Path('/home/viplab/data/thermal_slam_bags/ros2/results/{}'.format(self.bag_name))
        self.bridge = CvBridge()
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        self.reader = AnyReader([self.bag_path], default_typestore=typestore)
        # self.thermal_img_topic = '/optris/thermal_image'
        self.window_pose_topic = "/window_pose_array"
        self.window_gt_topic = "/gt_pose_array"
        self.point_marker_topic = "/point_marker"
        self.grad_img_topic = "/fsvio_ros2/grad_image"
        self.save_dir = "sfm_figs_150"
        self.topic_list = [self.window_pose_topic, self.point_marker_topic, self.window_gt_topic, self.grad_img_topic]
        
    def read_bag(self):
        point_sets = []
        pose_sets = []
        sfm_ts_list = []
        grad_img_list = []
        grad_img_ts_list = []   
        self.reader.open()
        connections = [x for x in self.reader.connections if x.topic in self.topic_list]
        for connection, timestamp, rawdata in self.reader.messages(connections=connections):
            msg = self.reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == self.point_marker_topic:
                pts = []
                for pt in msg.points:
                    pts.append([pt.x, pt.y, pt.z])
                point_sets.append(np.array(pts))
                sfm_ts_list.append(timestamp)
            elif connection.topic == self.window_pose_topic:
                poses = []
                for pose in msg.poses:
                    poses.append([pose.position.x, pose.position.y, pose.position.z, 
                                  pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
                pose_sets.append(np.array(poses))
            elif connection.topic == self.grad_img_topic:
                grad_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                grad_img_list.append(grad_img)
                grad_img_ts_list.append(timestamp)

        for idx in range(len(point_sets)):
            # print(idx)
            # self.plot_3d_points(point_sets[idx], pose_sets[idx])
            pass
        # sfm1: 37 39
        # sfm3: 20 60
        # sfm4: 27 28 30 49
        # clip 400: 32
        # clip 100: 33
        clip_idx = {}
        opt_100 = [72, 100, 102, 103, 104, 114]
        # idx_set = [i for i in range(len(point_sets))]
        idx_set = [self.name_idx_dict[self.bag_name]]
        for sfm_idx in idx_set:
            print("sfm idx: ", sfm_idx)
            self.plot_3d_points(point_sets[sfm_idx], pose_sets[sfm_idx], sfm_idx)
            # print(sfm_ts_list[37])
            # print(grad_img_ts_list[100])
            closest_idx = closest(grad_img_ts_list, sfm_ts_list[sfm_idx])
            # cv2.imwrite("xyz_clip/{}/grad_img_{}.png".format(self.bag_name, sfm_idx), grad_img_list[closest_idx-10])
        self.reader.close()

    def plot_3d_points(self, pts, poses, idx):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # Plot points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label='Points', c='b', s=4.0)

        # Plot poses as three axes (orientation)
        x_offset = np.mean(poses[:,0])
        # print("pose shape: ", poses.shape)
        # print("x offset: ", x_offset)
        for pose in poses:
            x, y, z = pose[:3]  # Position of the pose
            x = x - x_offset
            qw, qx, qy, qz = pose[3:]  # Orientation of the pose
            rot = Rot.from_quat([qx, qy, qz, qw])
            rot_m = rot.as_matrix()
            x_axis = rot_m @ np.array([[0.5, 0, 0]]).T
            y_axis = rot_m @ np.array([[0, 0.5, 0]]).T
            z_axis = rot_m @ np.array([[0, 0, 0.5]]).T
            ax.quiver(x, y, z, x_axis[0,0], x_axis[1,0], x_axis[2,0], color='r')
            ax.quiver(x, y, z, y_axis[0,0], y_axis[1,0], y_axis[2,0], color='g')
            ax.quiver(x, y, z, z_axis[0,0], z_axis[1,0], z_axis[2,0], color='b')

        # Set labels and title
        ax.set_xlim3d([-1.2, 1.2])
        ax.set_ylim3d([0, 2.4])
        ax.set_zlim3d([-0.8, 0.8])
        ax.set_box_aspect((1, 1, 2/3))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SFM Optimization 400ms')
        ax.view_init(elev=30, azim=-60)

        # Show the plot
        plt.legend()
        # plt.savefig("xyz_clip/{}/3d_plot_{}.png".format(self.bag_name, idx))
        plt.show()
        pass


if __name__ == '__main__':
    flow_from_bag = FlowFromBag()
    flow_from_bag.read_bag()
    