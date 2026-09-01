# Steps:

git clone --recurse-submodules https://github.com/Foresteh/gaitways_deploy_sean.git ~/ros2_ws

xhost +local:docker

docker pull brytsknguyen/gaitways_docker

docker run -it --name gaitways_docker_sean --network host --gpus all -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/ros2_ws:/root/ros2_ws --device=/dev/input -w /root brytsknguyen/gaitways_docker

cd /root/ros2_ws

source /opt/ros/humble/setup.bash

colcon build --symlink-install

source install/setup.bash

ros2 launch gaitways_deploy_sean gaitways_launch.py
