#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid


def grid_to_gray(grid: np.ndarray) -> np.ndarray:
    """
    OccupancyGrid values:
      -1 unknown
       0 free
     1..100 occupied probability/cost
    Output: uint8 grayscale image.
    """
    img = np.empty_like(grid, dtype=np.uint8)

    unknown = (grid < 0)
    known = ~unknown

    img[unknown] = 127  # mid-gray for unknown

    # map 0..100 -> 255..0 (white free, black occupied)
    g = np.clip(grid[known], 0, 100).astype(np.float32)
    img[known] = (255.0 * (1.0 - g / 100.0)).astype(np.uint8)

    return img


def grid_to_cost_rgb(grid: np.ndarray) -> np.ndarray:
    """
    Colored visualization:
      unknown -> gray
      free (0) -> white
      higher values -> red-ish, darkest for 100
    Output: uint8 RGB image.
    """
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    unknown = (grid < 0)
    known = ~unknown

    rgb[unknown] = (127, 127, 127)

    g = np.clip(grid[known], 0, 100).astype(np.float32) / 100.0

    # free -> white
    # occupied -> dark red
    # simple blend: white*(1-g) + red*(g)
    white = np.array([255, 255, 255], dtype=np.float32)
    red = np.array([180, 0, 0], dtype=np.float32)
    col = (white * (1.0 - g[:, None]) + red * g[:, None]).astype(np.uint8)

    rgb.reshape(-1, 3)[known.reshape(-1)] = col
    return rgb


class OccupancyGridSaver(Node):
    def __init__(self, topic: str, out_dir: str, every: int, mode: str, flip_y: bool):
        super().__init__("occupancy_grid_saver")

        self.topic = topic
        self.out_dir = out_dir
        self.every = max(1, every)
        self.mode = mode
        self.flip_y = flip_y
        self.count = 0

        os.makedirs(self.out_dir, exist_ok=True)

        # Match common map QoS: RELIABLE + TRANSIENT_LOCAL (latched map)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(OccupancyGrid, self.topic, self.cb, qos)
        self.get_logger().info(
            f"Saving OccupancyGrid from {self.topic} to {self.out_dir} (every={self.every}, mode={self.mode})"
        )

    def cb(self, msg: OccupancyGrid):
        self.count += 1
        if (self.count % self.every) != 0:
            return

        w = int(msg.info.width)
        h = int(msg.info.height)
        data = np.array(msg.data, dtype=np.int16)
        if data.size != w * h:
            self.get_logger().warn(f"Bad grid size: got {data.size}, expected {w*h}")
            return

        grid = data.reshape((h, w))  # row-major
        if self.flip_y:
            grid = np.flipud(grid)   # RViz-like orientation

        if self.mode == "gray":
            img = grid_to_gray(grid)
            pil = Image.fromarray(img, mode="L")
        else:
            rgb = grid_to_cost_rgb(grid)
            pil = Image.fromarray(rgb, mode="RGB")

        stamp = msg.header.stamp
        stamp_str = f"{stamp.sec}.{stamp.nanosec:09d}"
        frame = msg.header.frame_id.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(self.out_dir, f"occ_{ts}_rostime_{stamp_str}_frame_{frame}_{w}x{h}.png")

        pil.save(fname)
        self.get_logger().info(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/occupancy_grid")
    parser.add_argument("--out-dir", default="./occ_dump")
    parser.add_argument("--every", type=int, default=10, help="Save every Nth message")
    parser.add_argument("--mode", choices=["gray", "cost"], default="cost")
    parser.add_argument("--flip-y", action="store_true", help="Flip vertically to match RViz orientation")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = OccupancyGridSaver(args.topic, args.out_dir, args.every, args.mode, args.flip_y)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
