#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose

from tf2_ros import Buffer, TransformListener

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


def quat_to_rot(qx, qy, qz, qw):
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float64)


class CloudToGridSimple(Node):
    '''
    Fast occupancy grid:
      - marks ONLY occupied cells (no ray tracing/free-space)
      - accumulates over time
      - publishes nav_msgs/OccupancyGrid on /map

    Requires a TF chain: map -> <cloud_frame>
    '''

    def __init__(self):
        super().__init__("cloud_to_grid_simple")

        # Robust sim time set
        try:
            self.declare_parameter("use_sim_time", True)
        except ParameterAlreadyDeclaredException:
            pass
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # Inputs
        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.cloud_topic = self.declare_parameter("cloud_topic", "/R1/camera/points").value

        # Map settings
        self.resolution = float(self.declare_parameter("resolution", 0.20).value)   # coarse for speed
        self.size_x_m = float(self.declare_parameter("size_x_m", 120.0).value)
        self.size_y_m = float(self.declare_parameter("size_y_m", 120.0).value)
        self.auto_origin = bool(self.declare_parameter("auto_origin", True).value)
        self.origin_x = float(self.declare_parameter("origin_x", 0.0).value)
        self.origin_y = float(self.declare_parameter("origin_y", 0.0).value)
        self.origin_set = not self.auto_origin

        # Filters (in MAP frame after transform)
        self.z_min = float(self.declare_parameter("z_min", -10.0).value)
        self.z_max = float(self.declare_parameter("z_max",  50.0).value)
        self.r_min = float(self.declare_parameter("range_min", 0.5).value)
        self.r_max = float(self.declare_parameter("range_max", 20.0).value)

        # Downsample
        self.stride = int(self.declare_parameter("stride", 20).value)   # increase if slow
        self.max_points = int(self.declare_parameter("max_points", 20000).value)

        # Accumulation
        self.hit_cap = int(self.declare_parameter("hit_cap", 30).value)
        self.publish_every = int(self.declare_parameter("publish_every", 3).value)
        self._cloud_count = 0

        self.width = int(round(self.size_x_m / self.resolution))
        self.height = int(round(self.size_y_m / self.resolution))
        self.hits = np.zeros((self.height, self.width), dtype=np.uint8)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.cb, sub_qos)
        self.pub = self.create_publisher(OccupancyGrid, "/map", pub_qos)

        self._last_stamp = None
        self.create_timer(1.0, self._timer_publish)

        self.get_logger().info(
            f"OccupancyGrid from {self.cloud_topic} -> /map, {self.width}x{self.height} @ {self.resolution} m"
        )

    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def publish(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame

        info = MapMetaData()
        info.resolution = float(self.resolution)
        info.width = self.width
        info.height = self.height

        origin = Pose()
        origin.position.x = float(self.origin_x)
        origin.position.y = float(self.origin_y)
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        info.origin = origin
        msg.info = info

        # Unknown: -1; Occupied: 100 if hit>0
        grid = np.full((self.height, self.width), -1, dtype=np.int8)
        grid[self.hits > 0] = 100
        msg.data = grid.flatten(order="C").tolist()
        self.get_logger().info(f"msg is published")
        self.pub.publish(msg)

    def cb(self, msg: PointCloud2):
        # TF: map <- cloud_frame
        self.get_logger().info(f"msg is got in cb")
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, msg.header.stamp, timeout=Duration(seconds=0.2)
            )
            self.get_logger().info(f" try tf_buffer, msg.header.frame_id: {msg.header.frame_id}")
        except Exception:
            # fallback: latest
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, msg.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.2)
                )
                self.get_logger().info(f"fallback tf_buffer, msg.header.frame_id: {msg.header.frame_id}")
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed {self.map_frame}<-{msg.header.frame_id}: {e}")
                return

        self.get_logger().info("please print something")
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)

        # Set map origin once (center map around first pose)
        if (not self.origin_set) and self.auto_origin:
            self.origin_x = float(trans[0] - 0.5 * self.size_x_m)
            self.origin_y = float(trans[1] - 0.5 * self.size_y_m)
            self.origin_set = True
            self.get_logger().info(f"Auto origin set: origin_x={self.origin_x:.2f}, origin_y={self.origin_y:.2f}")

        # Read points with stride (and hard cap)
        pts = []
        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            self.get_logger().info(f"for i, p read points")
            if self.stride > 1 and (i % self.stride) != 0:
                continue
            pts.append(p)
            if len(pts) >= self.max_points:
                break
        if not pts:
            return

        pts_np = np.asarray(pts, dtype=np.float64)     # Nx3 cloud frame
        pts_map = (pts_np @ R.T) + trans               # Nx3 map frame

        # Filter
        dx = pts_map[:, 0] - trans[0]
        dy = pts_map[:, 1] - trans[1]
        rng = np.sqrt(dx*dx + dy*dy)
        self.get_logger().info(f"print in range!")
        m = (rng >= self.r_min) & (rng <= self.r_max) & (pts_map[:, 2] >= self.z_min) & (pts_map[:, 2] <= self.z_max)
        pts_map = pts_map[m]
        if pts_map.shape[0] == 0:
            return

        # Mark occupied
        for x, y, _z in pts_map:
            self.get_logger().info("x, y, z = {x}, {y}, {z}".format(x=x, y=y, z=_z))
            gx, gy = self.world_to_grid(x, y)
            if not self.in_bounds(gx, gy):
                continue
            v = int(self.hits[gy, gx])
            if v < self.hit_cap:
                self.hits[gy, gx] = v + 1

        # Publish periodically
        self._cloud_count += 1
        if (self._cloud_count % self.publish_every) == 0:
            self.get_logger().info(f"Cloud count: {self._cloud_count}")
            self.publish(msg.header.stamp)


def main():
    rclpy.init()
    node = CloudToGridSimple()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
