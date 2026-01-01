import copy
from typing import Optional

import rclpy
from builtin_interfaces.msg import Time
from rclpy.parameter import Parameter
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from pytictoc import TicToc


class BagDepthNode(Node):
    def __init__(self, drone_id: str = "R1"):
        super().__init__('bag_depth_node')
        self.camera_info_template = None
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.t = TicToc()
        self.drone_id = drone_id
        self.height = None
        self.width = None
        self.bridge = CvBridge()


        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # 1. Subscriber: Listen to the rosbag image stream
        self.img_sub = self.create_subscription(
            Image,
            f'/{self.drone_id}/camera/image_raw',  # Change this to match your bag's topic
            self.image_callback,
            camera_qos)

        # 2. Publishers: Output depth and info for Nvblox/VSLAM
        self.depth_pub = self.create_publisher(Image, f'/{self.drone_id}/camera/depth', camera_qos)
        self.info_pub = self.create_publisher(CameraInfo, f'/{self.drone_id}/camera/depth/camera_info', camera_qos)

        # Setup AI Model
        model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        self.depth_anything = DepthAnythingV2(**model_config)
        self.depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth'))
        self.depth_anything.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

        self.get_logger().info('Bag Depth Node Started. Waiting for images...')

    def intrinsic_from_fov(self,  hfov_deg=130, vfov_deg=90, half_pixel=True):
        theta_x = np.deg2rad(hfov_deg)
        theta_y = np.deg2rad(vfov_deg)

        fx = self.width / (2.0 * np.tan(theta_x / 2.0))
        fy = self.height / (2.0 * np.tan(theta_y / 2.0))

        if half_pixel:
            cx = (self.width - 1) / 2.0
            cy = (self.height - 1) / 2.0
        else:
            cx = self.width / 2.0
            cy = self.height / 2.0

        K = [fx, 0.0, cx,
             0.0, fy, cy,
             0.0, 0.0, 1.0]

        return K

    def get_camera_info(self, frame_id: str = "camera", stamp: Optional[Time] = None,
                        distortion_model: str = "plumb_bob",
                        ) -> CameraInfo:

        if self.camera_info_template is None:
            self.create_camera_info_template(stamp, frame_id, distortion_model)
            msg = self.camera_info_template
        else:
            msg = copy.deepcopy(self.camera_info_template)
            msg.header.stamp = stamp
            msg.header.frame_id = frame_id

        return msg

    def create_camera_info_template(self, stamp, frame_id, distortion_model):
        K = self.intrinsic_from_fov()
        K_list = list(K)
        if len(K_list) == 3 and hasattr(K_list[0], "__iter__"):
            K_list = [float(v) for row in K_list for v in row]

        if len(K_list) != 9:
            raise ValueError("K must contain 9 elements (3x3 matrix)")

        fx = K_list[0]
        fy = K_list[4]
        cx = K_list[2]
        cy = K_list[5]

        self.camera_info_template = CameraInfo()
        if stamp is not None:
            self.camera_info_template.header.stamp = stamp
        self.camera_info_template.header.frame_id = frame_id

        self.camera_info_template.width = self.width
        self.camera_info_template.height = self.height

        # Intrinsic matrix
        self.camera_info_template.k = K_list

        # Ideal camera: no distortion
        self.camera_info_template.distortion_model = distortion_model
        self.camera_info_template.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # 5 coeffs is common, can also use 0-length

        # Rectification matrix: identity (no stereo/rectification)
        self.camera_info_template.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # Projection matrix P (3x4), for monocular camera: K with Tx=0
        self.camera_info_template.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        self.t.tic()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Inference
        depth = self.depth_anything.infer_image(cv_image)

        dmin = float(depth.min())
        dmax = float(depth.max())
        den = max(dmax - dmin, 1e-6)

        depth_normalized = (depth - dmin) / den * 10.0  # pseudo-meters 0..10

        depth_msg = self.bridge.cv2_to_imgmsg(depth_normalized.astype(np.float32), encoding="32FC1")
        stamp_now = self.get_clock().now().to_msg()

        depth_msg.header = msg.header
        depth_msg.header.stamp = stamp_now
        depth_msg.header.frame_id = "main_camera_link"
        self.depth_pub.publish(depth_msg)
        self.t.toc()
        self.width = cv_image.shape[1]
        self.height = cv_image.shape[0]
        info_msg = self.get_camera_info(frame_id=depth_msg.header.frame_id, stamp=stamp_now)
        self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BagDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()