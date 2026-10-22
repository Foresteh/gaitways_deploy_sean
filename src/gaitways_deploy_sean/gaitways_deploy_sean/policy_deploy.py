#!/usr/bin/env python3
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

# import the message type for the joystick input
from sensor_msgs.msg import Joy

# import the message type for the UnitreeGo2 controls
from unitree_go.msg import LowCmd # type: ignore
from unitree_go.msg import LowState # type: ignore

# import the message type for Unitree API requests (used to release sport mode)
from unitree_api.msg import Request # type: ignore

from .gaitways_policy import DEFAULT_JOINT_ANGLES, REST_JOINT_ANGLES, GaitwaysPolicy, make_lowcmd


class PolicyDeploy(Node):

    def __init__(self):
        super().__init__('policy_deploy')

        self.policy = GaitwaysPolicy(str(Path(__file__).parent / 'model_29999.pt'))

        self._latest_commands = [0.0] * 12
        self._latest_commands[6] = 3.0   # gait_frequency
        self._latest_commands[7] = 0.5   # gait_phase
        self._latest_commands[11] = 0.0  # gait_id: 0 = trot

        self.last_run = 0.0
        self.on = False          # Select turns the policy on/off
        self.sitting = True      # robot starts lying down
        self.last_select = 0     # so holding Select only flips once

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

        # Publish to the motion switcher to release Unitree's sport mode (real robot only)
        self.release_publisher = self.create_publisher(
            Request,
            '/api/motion_switcher/request',
            10
        )

        # Wait up to 1 s for the robot to connect, then send the release request once
        for _ in range(10):
            if self.release_publisher.get_subscription_count() > 0:
                break
            time.sleep(0.1)
        release = Request()
        release.header.identity.api_id = 1003  # ReleaseMode
        self.release_publisher.publish(release)

    def joy_callback(self, msg):
        if msg.buttons[6] and not self.last_select:  # Back/Select
            self.on = not self.on
        self.last_select = msg.buttons[6]

        self._latest_commands[0] = float(msg.axes[1])  # forward/back
        self._latest_commands[1] = float(msg.axes[0])  # left/right
        self._latest_commands[2] = float(msg.axes[3])  # turn
        self._latest_commands[4] = 0.4 * msg.axes[8]  # D-pad up/down: pitch
        self._latest_commands[5] = 0.2 * msg.axes[7]  # D-pad left/right: roll

        if msg.buttons[0]:    # A: trot
            self._latest_commands[11] = 0.0
        elif msg.buttons[1]:  # B: pronk
            self._latest_commands[11] = 1.0
        elif msg.buttons[2]:  # X: pace
            self._latest_commands[11] = 2.0
        elif msg.buttons[3]:  # Y: bound
            self._latest_commands[11] = 3.0

    def state_callback(self, msg):
        if self.on == self.sitting:  # robot isn't in the pose Select asked for
            self.move(msg, DEFAULT_JOINT_ANGLES if self.on else REST_JOINT_ANGLES)
            self.sitting = not self.on
            return
        if not self.on:
            return
        now = time.monotonic()
        if now - self.last_run < 0.02:
            return
        self.last_run = now

        cmd = self.policy.build_lowcmd(msg.motor_state, msg.imu_state, self._latest_commands)
        self.publisher.publish(cmd)

    def move(self, msg, pose):
        # Move from the current pose to `pose` over 3 s
        start = [msg.motor_state[i].q for i in range(12)]
        for step in range(1, 151):
            a = step / 150
            a = a * a * (3 - 2 * a)  # ease in and out
            self.publisher.publish(make_lowcmd([s + a * (p - s) for s, p in zip(start, pose)], 50.0))
            time.sleep(0.02)
        self.policy = GaitwaysPolicy(str(Path(__file__).parent / 'model_29999.pt'))  # fresh memory for the next walk


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

