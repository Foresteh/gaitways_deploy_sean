from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start the MuJoCo simulator (stands in for the real robot)
        ExecuteProcess(
            cmd=['bash', 'run_go2_flat_sim.sh'],
            cwd='/root/ros2_ws/src/gaitways_deploy_sean',
            output='screen',
        ),

        Node(
            package='joy',
            executable='joy_node',
            name='joy_node'
        ),
        Node(
            package='gaitways_deploy_sean',
            executable='policy_deploy',
            name='policy_deploy'
        ),
    ])
