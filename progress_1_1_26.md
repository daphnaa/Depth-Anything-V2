# progress.md — ROS 2 bag → DepthAnything → PointCloud2 → (next) OccupancyGrid

Date: 2026-01-01

This doc summarizes today's progress and the **known-good configuration** that produces a live point cloud (`/R1/camera/points`) from a ROS 2 bag using DepthAnythingV2 + `depth_image_proc`.

---

## 1) Data in the bag

### Bag A rosbag2_2025_12_31-16_35_45 (images + camera info + IMU)
Topics (example):
- `/R1/camera/image_raw` (`sensor_msgs/Image`)
- `/R1/camera/camera_info` (`sensor_msgs/CameraInfo`)
- `/R1/visual_slam/imu` (`sensor_msgs/Imu`)

### Bag B rosbag2_2025_12_31-15_30_23 (includes FCU state + TF + images)
Topics (example):
- `/R1/fcu/state` (`fcu_driver_interfaces/msg/UAVState`) — contains `position` + `azimuth`
- `/Rooster_1/tf` (`tf2_msgs/TFMessage`) — **static** camera extrinsics (e.g. `base_link -> main_camera_link`)
- `/R1/camera/image_raw`, `/R1/camera/camera_info`

**Notes**
- `/Rooster_1/tf` being “static” is expected: it usually encodes camera mounting/extrinsics, not drone motion.
- Drone motion must come from `/R1/fcu/state` (or a SLAM/odom source).

---

## 2) ROS 2 env / CycloneDDS

### Goal
Use ROS domain **1** and CycloneDDS.

### Common failure encountered
CycloneDDS XML pinned to a non-existent interface IP:
- Example error: `192.168.131.24: does not match an available interface`
- Also: `NetworkInterfaceAddress` is deprecated in CycloneDDS config.

### Fix
Use a minimal CycloneDDS config (auto-pick interface) or update to the new `<Interfaces>` style.

Minimal `cyclonedds.xml` example:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
```

### RMW library missing (earlier)
If you get:
`librmw_cyclonedds_cpp.so: cannot open shared object file`
Install:
```bash
sudo apt update
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

---

## 3) GPU / PyTorch / DepthAnything

### Symptom
PyTorch warning:
`CUDA initialization: CUDA unknown error ...`

### Resolution
CUDA started working after environment cleanup and confirming torch can init CUDA:
```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Example result:
- `torch 2.9.1+cu128`
- `cuda avail True`
- GPU: `NVIDIA GeForce RTX 3070 Laptop GPU`

### Key code fix (ensure inference runs on GPU)
DepthAnything inference must move inputs to the model device:
```python
device = next(self.parameters()).device
image = image.to(device)
```

Your `infer_image()` ends with:
```python
return depth.cpu().numpy()
```
which is fine (it brings result back to CPU for publishing).

---

## 4) Depth → PointCloud2 pipeline (WORKING)

### Components
1. **Bag playback** (with `/clock`)
2. **DepthAnything node**: subscribes RGB, publishes depth + depth/camera_info
3. **depth_image_proc**: converts depth + camera_info to PointCloud2
4. **RViz2**: visualizes point cloud

### Bag playback
Use sim time:
```bash
ros2 bag play <bag_dir> --clock --rate 0.05
```
(0.05 was used to reduce load while debugging.)

### DepthAnything publishes
- `/R1/camera/depth` (`sensor_msgs/Image`, encoding `32FC1`)
- `/R1/camera/depth/camera_info` (`sensor_msgs/CameraInfo`)

### depth_image_proc executable
Correct executable name:
```bash
ros2 pkg executables depth_image_proc
# includes: point_cloud_xyz_node
```

Run:
```bash
ros2 run depth_image_proc point_cloud_xyz_node --ros-args \
  --remap image_rect:=/R1/camera/depth \
  --remap camera_info:=/R1/camera/depth/camera_info \
  --remap points:=/R1/camera/points
```

### PointCloud2 header/frame
Confirmed:
- `/R1/camera/points` `frame_id: main_camera_link`

---

## 5) QoS issues (RELIABILITY mismatch) and fix

### Symptom
Warnings like:
- `requesting incompatible QoS ... RELIABILITY_QOS_POLICY`
- no messages delivered

### Diagnosis
`ros2 topic info -v` showed:
- `PointCloudXyzNode` subscribes to depth topics as **RELIABLE**
- RViz often subscribes as **BEST_EFFORT**
- `PointCloudXyzNode` publishes `/R1/camera/points` as **BEST_EFFORT`

### Practical fix used
Make the DepthAnything publishers **RELIABLE** so `PointCloudXyzNode` can subscribe:
- Publish `/R1/camera/depth` and `/R1/camera/depth/camera_info` with RELIABLE QoS.
- RViz can remain BEST_EFFORT for visualization.

Helpful commands:
```bash
ros2 topic info /R1/camera/depth -v
ros2 topic info /R1/camera/depth/camera_info -v
ros2 topic info /R1/camera/points -v
```

---

## 6) TF status and what it means

### Observed
- `/Rooster_1/tf` contains **static** transform like:
  `base_link -> main_camera_link` with constant translation (e.g. ~0.13m, 0.0m, 0.014m)
- Attempting `tf2_echo map base_link` failed because `map` frame did not exist / tree disconnected.

### Key takeaway
- Static camera TF is fine (extrinsics).
- **Drone motion TF** is missing and must be generated from `/R1/fcu/state` (or SLAM/odom).

---

## 7) Occupancy grid (attempted; blocked by TF tree)

### Goal
Build an accumulating `nav_msgs/OccupancyGrid` (/map) from `/R1/camera/points`.

### Blocker encountered
`cloud_to_occupancy_grid` warnings:
- `Could not find a connection between 'map' and 'main_camera_link'`
- `Tf has two or more unconnected trees`
- `python3 cloud_to_occupancy_grid.py --ros-args -p z_min:=-5.0 -p z_max:=50.0 -p stride:=10`

This happens because TF did not have a connected chain:
`map -> ... -> main_camera_link`

### What’s needed next
Create a connected TF chain:

**Recommended approach**
1. Publish dynamic TF: `map -> base_link` from `/R1/fcu/state`
2. Keep static TF: `base_link -> main_camera_link` (from `/Rooster_1/tf` or static publisher)
3. Then occupancy node can transform points to `map` and accumulate.

If bag TF topic is `/Rooster_1/tf`, remap it to `/tf` during play:
```bash
ros2 bag play <bag_dir> --clock --ros-args --remap /Rooster_1/tf:=/tf
```
(Use `--remap`, not `-r` which is playback rate.)

---

## 8) Known-good “points visible in RViz” run order

Terminal 1 (env + bag):
```bash
source ~/ros2_env_domain1_cyclone.sh
ros2 bag play <bag_dir> --clock --rate 0.05
```

Terminal 2 (DepthAnything node):
```bash
source ~/ros2_env_domain1_cyclone.sh
python3 bag_depth_processor.py
```

Terminal 3 (PointCloud conversion):
```bash
source ~/ros2_env_domain1_cyclone.sh
ros2 run depth_image_proc point_cloud_xyz_node --ros-args \
  --remap image_rect:=/R1/camera/depth \
  --remap camera_info:=/R1/camera/depth/camera_info \
  --remap points:=/R1/camera/points
```

Terminal 4 (RViz2):
- Fixed Frame: `main_camera_link` (or another available frame)
- Add: **PointCloud2** display → topic `/R1/camera/points`

---

## 9) Next actions (recommended)

1. **Publish motion TF** from `/R1/fcu/state`
   - `map -> base_link` with translation from `state.position`
   - yaw from `state.azimuth` with correct convention

2. Ensure TF chain is connected:
   - `ros2 run tf2_ros tf2_echo map main_camera_link`

3. Run the accumulating occupancy node:
   - transform `/R1/camera/points` into `map`
   - filter Z + range
   - update log-odds and publish `/map`

4. (Optional) Map saving:
   - `nav2_map_server` `map_saver_cli` once `/map` is stable.

---

## 10) Notes / pitfalls

- **Sync warnings** between depth and camera_info are common with low rates; ensure timestamps match (use the same `msg.header.stamp` when publishing depth and depth/camera_info).
- RViz may drop messages if queue is small; increase RViz queue size if needed.
- Start slow (`--rate 0.05`) until the pipeline is stable, then increase.
