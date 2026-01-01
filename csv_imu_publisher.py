import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import pandas as pd  # Ensure pandas is installed: pip install pandas


class CsvImuPublisher(Node):
    def __init__(self):
        super().__init__('csv_imu_publisher')
        self.publisher_ = self.create_publisher(Imu, '/visual_slam/imu', 10)

        # Load your CSV data
        self.df = pd.read_csv('your_imu_data.csv')
        self.current_row = 0

        # Publish at 100Hz (standard for IMU) or match your CSV rate
        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        if self.current_row >= len(self.df):
            self.get_logger().info('End of CSV reached.')
            return

        row = self.df.iloc[self.current_row]
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Fill with your CSV column names (example below)
        msg.linear_acceleration.x = float(row['accel_x'])
        msg.linear_acceleration.y = float(row['accel_y'])
        msg.linear_acceleration.z = float(row['accel_z'])
        msg.angular_velocity.x = float(row['gyro_x'])
        msg.angular_velocity.y = float(row['gyro_y'])
        msg.angular_velocity.z = float(row['gyro_z'])

        self.publisher_.publish(msg)
        self.current_row += 1


def main(args=None):
    rclpy.init(args=args)
    node = CsvImuPublisher()
    rclpy.spin(node)
    rclpy.shutdown()