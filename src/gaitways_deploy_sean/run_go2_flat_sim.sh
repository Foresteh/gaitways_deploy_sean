#!/usr/bin/env bash
set -euo pipefail

SIM_LAUNCHER="/root/Tool/unitree_mujoco/run_sim_with_unitree_dds.sh"
ROBOT_NAME="go2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_PATH="$SCRIPT_DIR/scene_flat.xml"
RUNTIME_SCENE_NAME="sean_scene.xml"
ROBOT_SCENE_DIR="/root/Tool/unitree_mujoco/unitree_robots/$ROBOT_NAME"

cp "$SCENE_PATH" "$ROBOT_SCENE_DIR/$RUNTIME_SCENE_NAME"

exec "$SIM_LAUNCHER" -r "$ROBOT_NAME" -s "$RUNTIME_SCENE_NAME"
