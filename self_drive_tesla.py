import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time

class TeslaAutopilot(Node):
    def __init__(self):
        super().__init__('tesla_autopilot_node')
        self.publisher_ = self.create_publisher(
            Bool,
            '/carla/ego_vehicle/enable_autopilot',
            10)
        time.sleep(1.0) 

    def enable(self):
        msg = Bool()
        msg.data = True 
        self.get_logger().info('Publishing True to /carla/ego_vehicle/enable_autopilot...')
        for _ in range(10):
            self.publisher_.publish(msg)
            time.sleep(0.1)
        self.get_logger().info('Command sent.')

def main(args=None):
    rclpy.init(args=args)
    node = TeslaAutopilot()
    node.enable()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
