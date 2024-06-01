import rosbag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the ROS2 bag file
bag_file_path = 'your_bag_file.bag'
bag = rosbag.Bag(bag_file_path)

# Initialize lists to store points and poses
points = []
poses = []

# Extract points and poses from the bag file
for topic, msg, t in bag.read_messages(topics=['/your_points_topic', '/your_poses_topic']):
    if topic == '/your_points_topic':
        # Assuming the points are stored in MARKER.POINTS format
        for point in msg.points:
            points.append([point.x, point.y, point.z])
    elif topic == '/your_poses_topic':
        # Assuming poses are stored as geometry_msgs.msg.PoseArray
        for pose in msg.poses:
            poses.append([pose.position.x, pose.position.y, pose.position.z])

# Close the bag file
bag.close()

# Convert lists to NumPy arrays for plotting
points_array = np.array(points)
poses_array = np.array(poses)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], label='Points', c='b')

# Plot poses as three axes (orientation)
for pose in poses_array:
    x, y, z = pose[:3]  # Position of the pose
    ax.quiver(x, y, z, pose[3], pose[4], pose[5], color='r', label='Poses')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Points and Poses')

# Show the plot
plt.legend()
plt.show()
