import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class VideoDepthNode(Node):
    def __init__(self):
        super().__init__('video_depth_node')
        self.bridge = CvBridge()
        self.i = 0

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # Publishers
        self.img_pub = self.create_publisher(Image, '/camera/image_raw', camera_qos)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect', camera_qos)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', camera_qos)

        # Setup AI Model (Stick to 'vits' for speed)
        model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        self.depth_anything = DepthAnythingV2(**model_config)
        self.depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth'))
        self.depth_anything.to('cuda').eval()
        self.prev_depth = None

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transforms()


        # Video Input
        self.cap = cv2.VideoCapture('assets/examples_video/output.mp4')
        self.timer = self.create_timer(0.05, self.timer_callback)  #

    def publish_static_transforms(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "camera_link"
        # Adjust these if your camera is offset on the drone
        t.transform.translation.x = 0.1
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    def timer_callback(self):
        ret, frame = self.cap.read()
        self.i += 1

        if not ret:
            self.get_logger().info('Video Finished')
            return

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[100:620, 50:1230] = 255
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        if self.i > 0:
            now = self.get_clock().now().to_msg()
            current_depth = self.depth_anything.infer_image(masked_frame)
            depth_min = current_depth.min()
            depth_max = current_depth.max()
            # Normalize to 0-1
            normalized_depth = (current_depth - depth_min) / (depth_max - depth_min)
            # Map to real-world meters (Example: 0.5m to 5m range)
            real_world_depth = normalized_depth * 5.0
            if self.prev_depth is not None:
                smoothed_depth = (real_world_depth * 0.9) + (self.prev_depth * 0.1)
            else:
                smoothed_depth = real_world_depth
            self.prev_depth = smoothed_depth

            # 2. Publish RGB
            img_msg = self.bridge.cv2_to_imgmsg(masked_frame, encoding="rgb8")
            img_msg.header.stamp = now  # Set sync time
            img_msg.header.frame_id = "camera_optical_frame"
            self.get_logger().info('color published')
            self.img_pub.publish(img_msg)

            # 3. Publish Depth (32FC1 is the standard for Nvblox)
            # depth should be float32 representing meters
            depth_msg = self.bridge.cv2_to_imgmsg(smoothed_depth.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = "camera_optical_frame"  # Use optical frame for Nvblox
            self.depth_pub.publish(depth_msg)

            # 4. Publish Fake Camera Info (Standard 720p example)
            info_msg = CameraInfo()
            info_msg.header.stamp = now
            info_msg.header.frame_id = "camera_optical_frame"
            info_msg.width = frame.shape[1]
            info_msg.height = frame.shape[0]
            info_msg.k = [550.0, 0.0, 640.0, 0.0, 550.0, 360.0, 0.0, 0.0, 1.0]
            info_msg.p = [550.0, 0.0, 640.0, 0.0, 0.0, 550.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VideoDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()