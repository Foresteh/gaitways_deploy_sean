#!/usr/bin/env python3
"""
Standalone GaitWays policy model: input/output only.

This file has ONE job: given the robot's current sensor readings and your
desired command (from joystick, script, whatever), run the trained model
and produce a command ready to publish to the robot. `step()` gives you raw
target joint angles ("numbers in, numbers out"); `build_lowcmd()` wraps that
plus the Unitree LowCmd/CRC packing so calling code just needs to pass in
the raw LowState fields and publish the result.

Extracted and simplified from gaitways_deploy's policy_runner.py and
observation_builder.py. Same model file (model_29999.pt), same math -- just
stripped of everything ROS/ros2-specific so it can be reused standalone.

Requires: torch

------------------------------------------------------------------------
HOW TO USE
------------------------------------------------------------------------

    policy = GaitwaysPolicy("model_29999.pt")

    # once per control step (e.g. 50 Hz):
    target_joint_angles = policy.step(
        base_angular_velocity=[0.0, 0.0, 0.0],   # from IMU (rad/s)
        projected_gravity=[0.0, 0.0, -1.0],      # from IMU (unit vector)
        current_joint_angles=[...12 values...],  # from motor encoders
        current_joint_velocities=[...12 values...],
        commands=[...12 values...],              # your joystick/desired command
    )

    # target_joint_angles is a list of 12 floats, one per leg joint, in the
    # SAME physical joint order you passed current_joint_angles in.

    # if you start/stop or teleport the robot, reset the model's memory:
    policy.reset()

------------------------------------------------------------------------
"""

import ctypes
import math
import struct
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from crccheck.crc import Crc32Mpeg2

from unitree_go.msg import LowCmd  # type: ignore

# ---------------------------------------------------------------------------
# Fixed sizes -- these come from how the model was trained. Do not change
# unless you are loading a differently-trained checkpoint.
# ---------------------------------------------------------------------------
OBSERVATION_DIM = 73            # total input size to the actor model
ESTIMATOR_OBSERVATION_DIM = 70  # input size to the velocity-estimator model
NUM_JOINTS = 12
NUM_COMMANDS = 12

ACTION_CLAMP_MIN = -10.0
ACTION_CLAMP_MAX = 10.0

# Scales applied to the raw `commands` you provide, before they go into the
# observation. Taken directly from gaitways_deploy's control_config.yaml.
COMMANDS_SCALE = [1.0, 1.0, 1.0, 2.0, 0.3, 0.3, 1.0, 1.0, 0.15, 1.0, 1.0, 1.0]

# Default joint angles / safety limits, from gaitways_deploy's
# control_config.yaml. Order matches `policy_order_indices` below.
DEFAULT_JOINT_ANGLES = [
    0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5,
]
HARD_LOWER_LIMITS = [
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227,
]
HARD_UPPER_LIMITS = [
    1.0472, 3.4907, -0.8378,
    1.0472, 3.4907, -0.8378,
    1.0472, 4.5379, -0.8378,
    1.0472, 4.5379, -0.8378,
]
# Maps "policy's internal joint order" -> "your physical joint order".
POLICY_ORDER_INDICES = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

# Maps LowCmd/LowState's motor array slot -> the same physical joint order=
DEPLOYMENT_ORDER_INDICES = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

DEFAULT_ACTION_SCALE = 0.25
MOTOR_KP = 25.0
MOTOR_KD = 0.5


def _rpy_to_projected_gravity(rpy: Sequence[float]) -> list[float]:
    roll, pitch, _yaw = rpy
    sin_p, cos_p = math.sin(pitch), math.cos(pitch)
    sin_r, cos_r = math.sin(roll), math.cos(roll)
    return [sin_p, -cos_p * sin_r, -cos_p * cos_r]


# ---------------------------------------------------------------------------
# Model definition (must match the architecture used at training time)
# ---------------------------------------------------------------------------

class _FiLM(nn.Module):
    """Lets a small 'gait id' signal modulate the memory layer's input."""

    def __init__(self, gait_code_dim: int, observation_dim: int) -> None:
        super().__init__()
        self.to_gb = nn.Linear(gait_code_dim, 2 * observation_dim)

    def forward(self, observation: torch.Tensor, gait_code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gb(gait_code).chunk(2, dim=-1)
        return (1.0 + gamma) * observation + beta


class _RecurrentMemory(nn.Module):
    """The actor's memory (LSTM) -- makes the model stateful across steps."""

    def __init__(self, *, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.hidden_states = None
        self.gait_emb = nn.Embedding(4, 2)
        self.film_in = _FiLM(2, input_size)

    # 2-bit binary code per gait id 
    # 0 trot -> [0,0], 1 pronk -> [0,1], 2 pace -> [1,0], 3 bound -> [1,1].
    _GAIT_BINARY = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    @classmethod
    def _gait_binary(cls, gait_id: torch.Tensor) -> torch.Tensor:
        return cls._GAIT_BINARY.to(gait_id.device)[gait_id]

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        gait_id = torch.round(observation[..., -1]).long()
        gait_code = self._gait_binary(gait_id)
        modulated_input = self.film_in(observation, gait_code)
        out, self.hidden_states = self.rnn(modulated_input.unsqueeze(0), self.hidden_states)
        return out.squeeze(0)

    def reset(self) -> None:
        self.hidden_states = None


class _Actor(nn.Module):
    """73 numbers in -> 12 raw actions out."""

    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU()
        self.actor = nn.Sequential(
            nn.Linear(512, 512), activation,
            nn.Linear(512, 256), activation,
            nn.Linear(256, 128), activation,
            nn.Linear(128, 12),
        )
        # `critic` and `std` exist in the checkpoint (used during training)
        # but are unused for inference. Kept so load_state_dict succeeds.
        self.critic = nn.Sequential(
            nn.Linear(512, 512), activation,
            nn.Linear(512, 256), activation,
            nn.Linear(256, 128), activation,
            nn.Linear(128, 1),
        )
        self.std = nn.Parameter(torch.ones(12))
        self.memory_a = _RecurrentMemory(input_size=OBSERVATION_DIM, hidden_size=512)
        self.memory_c = _RecurrentMemory(input_size=OBSERVATION_DIM, hidden_size=512)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.memory_a(observation))

    def reset(self) -> None:
        self.memory_a.reset()
        self.memory_c.reset()


class _EstimatorMemory(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.hidden_states = None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        out, self.hidden_states = self.rnn(observation.unsqueeze(0), self.hidden_states)
        return out.squeeze(0)

    def reset(self) -> None:
        self.hidden_states = None


class _Estimator(nn.Module):
    """70 numbers in -> 3 numbers out (guessed base linear velocity).

    Exists because a real robot has no direct sensor for its own forward/
    side/up velocity -- this network guesses it from other sensor data.
    """

    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU()
        self.memory_a = _EstimatorMemory(input_size=ESTIMATOR_OBSERVATION_DIM, hidden_size=256)
        self.latent_encoder = nn.Sequential(
            nn.Linear(256, 256), activation,
            nn.Linear(256, 128), activation,
        )
        self.latent_output = nn.Linear(128, 3)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        latent_input = self.memory_a(observation)
        latent = self.latent_encoder(latent_input)
        return self.latent_output(latent)

    def reset(self) -> None:
        self.memory_a.reset()


# ---------------------------------------------------------------------------
# Gait clock -- produces a repeating signal telling the model where each
# foot currently is in its walking cycle.
# ---------------------------------------------------------------------------

class _GaitClock:
    TROT, PRONK, PACE, BOUND = 0, 1, 2, 3

    def __init__(self, *, step_dt: float) -> None:
        self.step_dt = float(step_dt)
        self._t_global = 0.0

    def reset(self) -> None:
        self._t_global = 0.0

    def step(self, *, gait_frequency: float, gait_phase: float, gait_id: float) -> list[float]:
        gait_id = int(round(float(gait_id)))
        self._t_global = (self._t_global + self.step_dt * gait_frequency) % 1.0
        if gait_id == self.PRONK:
            theta1, theta2, theta3 = 0.0, 0.0, 0.0
        elif gait_id == self.TROT:
            theta1, theta2, theta3 = gait_phase, 0.0, 0.0
        elif gait_id == self.PACE:
            theta1, theta2, theta3 = 0.0, gait_phase, 0.0
        elif gait_id == self.BOUND:
            theta1, theta2, theta3 = 0.0, 0.0, gait_phase
        else:
            raise ValueError(f"Unsupported gait_id: {gait_id}")

        t_foot = [
            (self._t_global + theta1 + theta2 + theta3) % 1.0,
            (self._t_global + theta2) % 1.0,
            (self._t_global + theta3) % 1.0,
            (self._t_global + theta1) % 1.0,
        ]
        return [math.sin(2.0 * math.pi * t) for t in t_foot]


# ---------------------------------------------------------------------------
# The one class you actually use
# ---------------------------------------------------------------------------

class GaitwaysPolicy:
    """Loads the model once, then call `.step(...)` every control tick."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        default_joint_angles: Sequence[float] = DEFAULT_JOINT_ANGLES,
        hard_lower_limits: Sequence[float] = HARD_LOWER_LIMITS,
        hard_upper_limits: Sequence[float] = HARD_UPPER_LIMITS,
        policy_order_indices: Sequence[int] = POLICY_ORDER_INDICES,
        action_scale: float = DEFAULT_ACTION_SCALE,
        control_dt: float = 1.0 / 50.0,
        use_cuda: bool = True,
    ) -> None:
        self.default_joint_angles = [float(v) for v in default_joint_angles]
        self.hard_lower_limits = [float(v) for v in hard_lower_limits]
        self.hard_upper_limits = [float(v) for v in hard_upper_limits]
        self.policy_order_indices = [int(v) for v in policy_order_indices]
        self.action_scale = float(action_scale)

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.actor, self.estimator = self._load(Path(model_path), self.device)

        self.clock = _GaitClock(step_dt=control_dt)
        self._action_t = [0.0] * NUM_JOINTS
        self._action_t_minus_1 = [0.0] * NUM_JOINTS

    # -- public API ----------------------------------------------------

    def reset(self) -> None:
        """Call this whenever you start a new control session (the model
        has memory across steps, so old state should not carry over)."""
        self.actor.reset()
        self.estimator.reset()
        self.clock.reset()
        self._action_t = [0.0] * NUM_JOINTS
        self._action_t_minus_1 = [0.0] * NUM_JOINTS

    def step(
        self,
        *,
        base_angular_velocity: Sequence[float],   # 3 values, rad/s
        projected_gravity: Sequence[float],       # 3 values, unit vector
        current_joint_angles: Sequence[float],    # 12 values, rad
        current_joint_velocities: Sequence[float],  # 12 values, rad/s
        commands: Sequence[float],                # 12 values -- see NOTE below
    ) -> list[float]:
        """Run one control step. Returns 12 target joint angles (rad), in
        the same physical joint order you passed the inputs in.

        NOTE on `commands`: this is the same 12-value vector
        gaitways_deploy's GaitCommand message carries. Three of its slots
        are NOT just "raw command values" fed to the model -- they also
        drive the internal gait clock:
            commands[6]  = gait_frequency
            commands[7]  = gait_phase
            commands[11] = gait_id (0=trot, 1=pronk, 2=pace, 3=bound)
        All 12 values (including those three) are still scaled by
        COMMANDS_SCALE and placed in the observation as normal.
        """

        observation = self._build_observation(
            base_angular_velocity=base_angular_velocity,
            projected_gravity=projected_gravity,
            current_joint_angles=current_joint_angles,
            current_joint_velocities=current_joint_velocities,
            commands=commands,
        )

        obs_tensor = torch.tensor([observation], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # The estimator guesses the robot's own linear velocity, since
            # there's no direct sensor for it. Its output replaces the
            # placeholder first 3 values in `observation`.
            estimated_velocity = self.estimator.act(obs_tensor[:, 3:]).squeeze(0)
            policy_observation = torch.cat(
                [estimated_velocity.unsqueeze(0), obs_tensor[:, 3:]], dim=1
            )
            raw_action = self.actor.act(policy_observation).squeeze(0)

        raw_action = [
            max(ACTION_CLAMP_MIN, min(ACTION_CLAMP_MAX, float(v)))
            for v in raw_action.detach().cpu().tolist()
        ]

        target_joint_angles = self._action_to_target_joint_angles(raw_action)

        self._action_t_minus_1 = self._action_t
        self._action_t = raw_action

        return target_joint_angles

    def build_lowcmd(self, motor_state, imu_state, commands: Sequence[float]) -> LowCmd:
        """Takes LowState's `motor_state`/`imu_state` fields plus a 12-value
        commands vector, runs the model, and returns a ready-to-publish
        LowCmd (CRC included)."""

        current_joint_angles = [0.0] * NUM_JOINTS
        current_joint_velocities = [0.0] * NUM_JOINTS
        for motor_idx in range(NUM_JOINTS):
            base_idx = DEPLOYMENT_ORDER_INDICES[motor_idx]
            current_joint_angles[base_idx] = float(motor_state[motor_idx].q)
            current_joint_velocities[base_idx] = float(motor_state[motor_idx].dq)

        target_joint_angles = self.step(
            base_angular_velocity=list(imu_state.gyroscope[0:3]),
            projected_gravity=_rpy_to_projected_gravity(imu_state.rpy[0:3]),
            current_joint_angles=current_joint_angles,
            current_joint_velocities=current_joint_velocities,
            commands=commands,
        )

        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        for motor_idx in range(NUM_JOINTS):
            base_idx = DEPLOYMENT_ORDER_INDICES[motor_idx]
            motor = cmd.motor_cmd[motor_idx]
            motor.mode = 0x01
            motor.q = float(target_joint_angles[base_idx])
            motor.dq = 0.0
            motor.kp = MOTOR_KP
            motor.kd = MOTOR_KD
            motor.tau = 0.0
        _fill_crc(cmd)
        return cmd

    # -- internals -------------------------------------------------------

    def _build_observation(
        self,
        *,
        base_angular_velocity: Sequence[float],
        projected_gravity: Sequence[float],
        current_joint_angles: Sequence[float],
        current_joint_velocities: Sequence[float],
        commands: Sequence[float],
    ) -> list[float]:
        commands = [float(c) for c in commands]

        # Joint position/velocity are re-ordered from your physical joint
        # order into the policy's internal order (same mapping used for
        # the output action), and compared against each joint's default.
        joint_pos_error = [0.0] * NUM_JOINTS
        joint_vel_scaled = [0.0] * NUM_JOINTS
        for policy_idx, base_idx in enumerate(self.policy_order_indices):
            joint_pos_error[policy_idx] = (
                float(current_joint_angles[base_idx]) - self.default_joint_angles[base_idx]
            )
            joint_vel_scaled[policy_idx] = 0.05 * float(current_joint_velocities[base_idx])

        commands_scaled = [c * s for c, s in zip(commands, COMMANDS_SCALE)]

        # The gait clock is driven by specific slots inside `commands`,
        # not by fixed config values -- this matches gaitways_deploy's
        # GaitCommand message layout.
        clock_inputs = self.clock.step(
            gait_frequency=commands[6],
            gait_phase=commands[7],
            gait_id=commands[11],
        )

        return (
            [0.0, 0.0, 0.0]                       # placeholder for base_lin_vel
            + [float(v) for v in base_angular_velocity]
            + [float(v) for v in projected_gravity]
            + joint_pos_error
            + joint_vel_scaled
            + self._action_t
            + self._action_t_minus_1
            + clock_inputs
            + commands_scaled
        )

    def _action_to_target_joint_angles(self, raw_action: list[float]) -> list[float]:
        target_joint_angles = list(self.default_joint_angles)
        for policy_idx, base_idx in enumerate(self.policy_order_indices):
            angle = self.default_joint_angles[base_idx] + self.action_scale * raw_action[policy_idx]
            angle = max(self.hard_lower_limits[base_idx], min(self.hard_upper_limits[base_idx], angle))
            target_joint_angles[base_idx] = angle
        return target_joint_angles

    @staticmethod
    def _load(model_path: Path, device: torch.device) -> tuple[_Actor, _Estimator]:
        checkpoint = torch.load(model_path.as_posix(), map_location=device)
        actor = _Actor()
        estimator = _Estimator()
        actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
        estimator.load_state_dict(checkpoint["estimator_state_dict"], strict=True)
        actor.to(device).eval()
        estimator.to(device).eval()
        return actor, estimator


# ---------------------------------------------------------------------------
# LowCmd CRC -- required part of the Unitree wire format, only used by
# GaitwaysPolicy.build_lowcmd(). Kept at the bottom because it is a
# self-contained byte-packing utility unrelated to the model.
# ---------------------------------------------------------------------------

class _RawMotorCmd(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint8), ("q", ctypes.c_float), ("dq", ctypes.c_float),
        ("tau", ctypes.c_float), ("kp", ctypes.c_float), ("kd", ctypes.c_float),
        ("reserve", ctypes.c_uint32 * 3),
    ]


class _RawBmsCmd(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint8), ("reserve", ctypes.c_uint8 * 3)]


class _RawLowCmd(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint8 * 2), ("level_flag", ctypes.c_uint8),
        ("frame_reserve", ctypes.c_uint8), ("sn", ctypes.c_uint32 * 2),
        ("version", ctypes.c_uint32 * 2), ("bandwidth", ctypes.c_uint16),
        ("motor_cmd", _RawMotorCmd * 20), ("bms_cmd", _RawBmsCmd),
        ("wireless_remote", ctypes.c_uint8 * 40), ("led", ctypes.c_uint8 * 12),
        ("fan", ctypes.c_uint8 * 2), ("gpio", ctypes.c_uint8),
        ("reserve", ctypes.c_uint32), ("crc", ctypes.c_uint32),
    ]


def _copy_into(raw: ctypes.Structure, msg: object) -> None:
    for name, field_type in raw._fields_:
        msg_value = getattr(msg, name)
        raw_value = getattr(raw, name)
        if issubclass(field_type, ctypes.Structure):
            _copy_into(raw_value, msg_value)
        elif issubclass(field_type, ctypes.Array):
            element_is_struct = issubclass(field_type._type_, ctypes.Structure)
            for i in range(len(raw_value)):
                if element_is_struct:
                    _copy_into(raw_value[i], msg_value[i])
                else:
                    raw_value[i] = msg_value[i]
        else:
            setattr(raw, name, msg_value)


def _fill_crc(msg: LowCmd) -> None:
    raw = _RawLowCmd()
    _copy_into(raw, msg)
    size = ctypes.sizeof(_RawLowCmd) - ctypes.sizeof(ctypes.c_uint32)
    data = ctypes.string_at(ctypes.byref(raw), size)
    words = struct.unpack(f"<{size // 4}I", data)
    msg.crc = Crc32Mpeg2.calc(struct.pack(f">{size // 4}I", *words))


if __name__ == "__main__":
    # Quick smoke test -- run this file directly with a model path to
    # confirm it loads and produces output.
    import sys

    if len(sys.argv) != 2:
        print("Usage: python gaitways_policy.py <path-to-model.pt>")
        raise SystemExit(1)

    policy = GaitwaysPolicy(sys.argv[1], use_cuda=False)
    commands = [0.0] * NUM_COMMANDS
    commands[6] = 3.0   # gait_frequency (must be > 0)
    commands[7] = 0.5   # gait_phase
    commands[11] = 0.0  # gait_id: 0=trot
    result = policy.step(
        base_angular_velocity=[0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        current_joint_angles=DEFAULT_JOINT_ANGLES,
        current_joint_velocities=[0.0] * NUM_JOINTS,
        commands=commands,
    )
    print("target_joint_angles:", result)
