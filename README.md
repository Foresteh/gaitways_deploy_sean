# gaitways_deploy_sean

Deploys the trained GaitWays locomotion policy on a Unitree Go2, either in the
MuJoCo simulator or on the real robot. Both paths run the same node
(`policy_deploy`), which subscribes to `/lowstate`, runs the policy, and
publishes `/lowcmd`. A joystick drives the velocity/gait commands.

---

## 1. Common setup (do this once, for sim *or* real robot)

### 1.1 Clone the workspace (with submodules)

```bash
git clone --recurse-submodules https://github.com/Foresteh/gaitways_deploy_sean.git ~/ros2_ws
```

If you already cloned without `--recurse-submodules`:

```bash
cd ~/ros2_ws && git submodule update --init --recursive
```

### 1.2 Allow the container to open GUI windows (needed for the sim viewer)

```bash
xhost +local:docker
```

### 1.3 Pull the development image

```bash
docker pull brytsknguyen/gaitways_docker
```

This image already contains ROS 2 Humble, the `unitree_go` / `unitree_api`
messages, CycloneDDS, and `unitree_mujoco` (installed at
`/root/Tool/unitree_mujoco`).

### 1.4 Start the container

```bash
docker run -it --name gaitways_docker_sean \
    --network host \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/ros2_ws:/root/ros2_ws \
    --device=/dev/input \
    -w /root \
    brytsknguyen/gaitways_docker
```

Notes:
- `--network host` is required so ROS 2 / DDS traffic reaches the simulator or
  the robot.
- `--device=/dev/input` passes the joystick through to the container.
- To open a second shell in a running container:
  `docker exec -it gaitways_docker_sean bash`
- To re-attach later after the container was stopped:
  `docker start -ai gaitways_docker_sean`
- To remove a old container:
  `docker rm gaitways_docker_sean`

### 1.5 Build the workspace (inside the container)

```bash
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Re-run `source install/setup.bash` in every new shell.


## 2. Run in simulation

The sim launch file starts the MuJoCo simulator (via
`run_go2_flat_sim.sh`, which loads `scene_flat.xml` on a flat plane), the
`joy_node`, and `policy_deploy`.

```bash
# inside the container, from /root/ros2_ws, with the workspace sourced
ros2 launch gaitways_deploy_sean sim_launch.py
```

You should see the MuJoCo viewer appear with the Go2 standing on flat ground.
Use the joystick to drive it. If the viewer does not appear, re-run
`xhost +local:docker` on the host and restart the container.
