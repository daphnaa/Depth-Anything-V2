## Isaac ROS VSLAM & nvblox Environment Summary 
## 29/12/2025

----

### 1.Environment Configuration
* Host System: Linux PC with NVIDIA GPU (x86_64).
* Docker Container: isaac_ros_dev_container. 
  * ```cd ~/workspaces/isaac_ros-dev && isaac-ros activate ```
  
* ROS 2 Middleware: rmw_cyclonedds_cpp.
* Sensor Input: Monocular RGB + Depth at 20 FPS (50ms frame interval).
* Tracking Status: vo_state: 1 (Healthy/Tracking established).

### 2. Terminal & Node Setup
#### Terminal 1: Isaac ROS Visual SLAM (Inside Container)

Command: 
```bash 
ros2 launch isaac_ros_visual_slam my_vslam.launch.py
```
Key Launch Parameters:
* num_cameras: 1 (Monocular mode).
* image_jitter_threshold_ms: 100.0 (To handle 20 FPS timing).
* rectified_images: False (Input is raw drone video).
* enable_imu_fusion: False (Currently using visual-only odometry).

#### Terminal 2: Static Transform Publisher (Inside Container)
Command: 
```bash 
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw -1.57 --pitch 0 --roll -1.57 --frame-id base_link --child-frame-id camera_optical_frame
```
Purpose: Aligns the drone's body frame (base_link) with the camera's optical frame.

#### Terminal 3: Sensor Feed & Depth Node (On Host)
* Nodes: video_depth_node.
* Input: MP4 Drone recording.
* Output: Publishes /camera/image_raw and /depth_anything/image.
* Important: Ensure RMW_IMPLEMENTATION=rmw_cyclonedds_cpp is set on the host.

#### Terminal 4: nvblox (Inside Container)
Command:
```bash
ros2 launch nvblox_examples_bringup nvblox.launch.py     mode:=static     camera:=realsense     global_frame:=odom     depth_topic:=/camera/depth/image_raw     color_topic:=/camera/image_raw     camera_info_topic:=/camera/camera_info
```
* Status: Pending.
* Blocker: Waiting for SLAM to move from its initial zero-pose to provide a non-static map -> base_link transform.

### 3. Current Troubleshooting Status
* The "Zero Odometry" Issue: The odo.txt output confirms that position and orientation are currently all 0.0.
* Observations Cloud: Currently empty, indicating 2D features are not yet being extracted from the video feed.
* Required Action: Monocular SLAM requires significant translational movement (not just rotation) or IMU fusion to solve for scale and begin publishing non-zero odometry.
* To automate the environment setup, you can create a script ros2.env:

```bash
#!/bin/bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
source install/setup.bash
echo "Isaac ROS Environment Ready"
```