#!/usr/bin/env python3
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from unitree_api.msg import Request # type: ignore
from unitree_go.msg import LowCmd, LowState # type: ignore

from .gaitways_policy import GaitwaysPolicy


class PolicyDeploy(Node):

    def __init__(self):
        super().__init__('policy_deploy')

        self.policy = GaitwaysPolicy(str(Path(__file__).parent / 'model_29999.pt'))

        self._latest_commands = [0.0] * 12
        self._latest_commands[6] = 3.0   # gait_frequency
        self._latest_commands[7] = 0.5   # gait_phase
        self._latest_commands[11] = 0.0  # gait_id: 0 = trot

        self.last_run = 0.0

        # Subscribe to the joystick input topic
        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )

        # Subscribe to the robot's low-level state feedback topic
        self.state_subscription = self.create_subscription(
                    LowState,
                    '/lowstate',
                    self.state_callback,
                    10
        )

        # Publish to LowCmd topic to control the UnitreeGo2 robot
        self.publisher = self.create_publisher(
            LowCmd,
            '/lowcmd',
            10
        )

        self.release_publisher = self.create_publisher(
            Request,
            '/api/motion_switcher/request',
            10
        )

        time.sleep(1)
        release = Request()
        release.header.identity.api_id = 1003  # ReleaseMode
        self.release_publisher.publish(release)

    def joy_callback(self, msg):
        self._latest_commands[0] = float(msg.axes[1])  # forward/back
        self._latest_commands[1] = float(msg.axes[0])  # left/right
        self._latest_commands[2] = float(msg.axes[3])  # turn

        if msg.buttons[0]:    # A: trot
            self._latest_commands[11] = 0.0
        elif msg.buttons[1]:  # B: pronk
            self._latest_commands[11] = 1.0
        elif msg.buttons[2]:  # X: pace
            self._latest_commands[11] = 2.0
        elif msg.buttons[3]:  # Y: bound
            self._latest_commands[11] = 3.0

    def state_callback(self, msg):
        now = time.monotonic()
        if now - self.last_run < 0.02:
            return
        self.last_run = now

        cmd = self.policy.build_lowcmd(msg.motor_state, msg.imu_state, self._latest_commands)
        self.publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyDeploy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
