#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
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


def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield x1, y1


class CloudToOccupancyGrid(Node):
    def __init__(self):
        super().__init__("cloud_to_occupancy_grid")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.cloud_topic = self.declare_parameter("cloud_topic", "/R1/camera/points").value

        # Map geometry
        self.resolution = float(self.declare_parameter("resolution", 0.10).value)
        self.size_x_m = float(self.declare_parameter("size_x_m", 80.0).value)
        self.size_y_m = float(self.declare_parameter("size_y_m", 80.0).value)

        # Auto-origin: center map on first sensor pose
        self.auto_origin = bool(self.declare_parameter("auto_origin", True).value)
        self.origin_x = float(self.declare_parameter("origin_x", 0.0).value)
        self.origin_y = float(self.declare_parameter("origin_y", 0.0).value)
        self.origin_set = not self.auto_origin

        # Filters (in MAP frame)
        self.z_min = float(self.declare_parameter("z_min", 0.15).value)
        self.z_max = float(self.declare_parameter("z_max", 2.50).value)
        self.r_min = float(self.declare_parameter("range_min", 0.3).value)
        self.r_max = float(self.declare_parameter("range_max", 12.0).value)

        # Log-odds accumulation
        self.l_hit = float(self.declare_parameter("log_odds_hit", 0.85).value)
        self.l_miss = float(self.declare_parameter("log_odds_miss", 0.40).value)
        self.l_min = float(self.declare_parameter("log_odds_min", -4.0).value)
        self.l_max = float(self.declare_parameter("log_odds_max",  4.0).value)

        self.stride = int(self.declare_parameter("stride", 6).value)  # take every Nth point

        self.width = int(round(self.size_x_m / self.resolution))
        self.height = int(round(self.size_y_m / self.resolution))
        self.logodds = np.zeros((self.height, self.width), dtype=np.float32)

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

        self.get_logger().info(
            f"Accumulating OccupancyGrid from {self.cloud_topic} -> /map, {self.width}x{self.height} @ {self.resolution} m"
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

        # logodds -> probability -> occupancy
        p = 1.0 - 1.0 / (1.0 + np.exp(self.logodds))
        grid = np.full((self.height, self.width), -1, dtype=np.int8)

        known = np.abs(self.logodds) > 0.05
        grid[known] = np.clip((p[known] * 100.0).round(), 0, 100).astype(np.int8)

        msg.data = grid.flatten(order="C").tolist()
        self.pub.publish(msg)

    def cb(self, msg: PointCloud2):
        # map <- cloud_frame at cloud stamp
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, msg.header.stamp, timeout=Duration(seconds=0.2)
            )
        except Exception:
            # fallback: latest
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, msg.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.2)
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed map<-{msg.header.frame_id}: {e}")
                return

        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float64)

        # sensor origin in map
        sx, sy, sz = trans
        if (not self.origin_set) and self.auto_origin:
            self.origin_x = sx - 0.5 * self.size_x_m
            self.origin_y = sy - 0.5 * self.size_y_m
            self.origin_set = True
            self.get_logger().info(f"Auto origin set: origin_x={self.origin_x:.2f}, origin_y={self.origin_y:.2f}")

        gsx, gsy = self.world_to_grid(sx, sy)
        if not self.in_bounds(gsx, gsy):
            self.publish(msg.header.stamp)
            return

        # Read points (downsample + skip NaNs)
        pts = []
        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if self.stride > 1 and (i % self.stride) != 0:
                continue
            pts.append(p)
        if not pts:
            self.publish(msg.header.stamp)
            return

        pts_np = np.asarray(pts, dtype=np.float64)  # Nx3 cloud frame
        pts_map = (pts_np @ R.T) + trans            # Nx3 map frame

        # Filters
        dx = pts_map[:, 0] - sx
        dy = pts_map[:, 1] - sy
        rng = np.sqrt(dx*dx + dy*dy)
        m = (rng >= self.r_min) & (rng <= self.r_max) & (pts_map[:, 2] >= self.z_min) & (pts_map[:, 2] <= self.z_max)
        pts_map = pts_map[m]
        if pts_map.shape[0] == 0:
            self.publish(msg.header.stamp)
            return

        # Update log-odds with ray tracing
        for x, y, z in pts_map:
            gx, gy = self.world_to_grid(x, y)
            if not self.in_bounds(gx, gy):
                continue

            for cx, cy in bresenham(gsx, gsy, gx, gy):
                if not self.in_bounds(cx, cy):
                    break
                if cx == gx and cy == gy:
                    break
                self.logodds[cy, cx] = np.clip(self.logodds[cy, cx] - self.l_miss, self.l_min, self.l_max)

            self.logodds[gy, gx] = np.clip(self.logodds[gy, gx] + self.l_hit, self.l_min, self.l_max)

        self.publish(msg.header.stamp)


def main():
    rclpy.init()
    node = CloudToOccupancyGrid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
