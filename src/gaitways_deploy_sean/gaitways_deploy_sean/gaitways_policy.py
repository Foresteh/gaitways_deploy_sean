import ctypes
import math
import struct
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from crccheck.crc import Crc32Mpeg2

from unitree_go.msg import LowCmd  # type: ignore


NUM_JOINTS = 12

COMMANDS_SCALE = [1.0, 1.0, 1.0, 2.0, 0.3, 0.3, 1.0, 1.0, 0.15, 1.0, 1.0, 1.0]

DEFAULT_JOINT_ANGLES = [
    -0.1, 0.9, -1.8,
    0.1, 0.9, -1.8,
    -0.1, 1.2, -1.8,
    0.1, 1.2, -1.8,
]
# Lying down pose (master's proning_joint_angles, in motor order)
REST_JOINT_ANGLES = [
    0.2, 1.3, -2.6,
    -0.2, 1.3, -2.6,
    -0.3, 1.3, -2.6,
    0.3, 1.3, -2.6,
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

# POLICY_TO_MOTOR[i] = motor slot of the policy's i-th joint.
POLICY_TO_MOTOR = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

DEFAULT_ACTION_SCALE = 0.25
MOTOR_KP = 25.0
MOTOR_KD = 0.5



def _rpy_to_projected_gravity(rpy: Sequence[float]) -> list[float]:
    roll, pitch, _yaw = rpy
    sin_p, cos_p = math.sin(pitch), math.cos(pitch)
    sin_r, cos_r = math.sin(roll), math.cos(roll)
    return [sin_p, -cos_p * sin_r, -cos_p * cos_r]


# --- Model definition (must match the architecture used at training time) ---

class _Memory(nn.Module):
    """LSTM memory -- makes the model stateful across steps."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.hidden_states = None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        output, self.hidden_states = self.rnn(observation.unsqueeze(0), self.hidden_states)
        return output.squeeze(0)


class _Estimator(nn.Module):
    """70 numbers in -> 3 numbers out (guessed base linear velocity).

    Exists because a real robot has no direct sensor for its own forward/
    side/up velocity -- this network guesses it from other sensor data.
    """

    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU()
        self.memory_a = _Memory(input_size=70, hidden_size=256)  # 70 = observation minus lin vel
        self.latent_encoder = nn.Sequential(
            nn.Linear(256, 256), activation,
            nn.Linear(256, 128), activation,
        )
        self.latent_output = nn.Linear(128, 3)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        latent_input = self.memory_a(observation)
        latent = self.latent_encoder(latent_input)
        return self.latent_output(latent)

class _FiLM(nn.Module):
    """Lets a small 'gait id' signal modulate the memory layer's input."""

    def __init__(self) -> None:
        super().__init__()
        self.to_gb = nn.Linear(2, 2 * 73)  # gait code -> gamma (scale), beta (shift)

    def forward(self, observation: torch.Tensor, gait_code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gb(gait_code).chunk(2, dim=-1)
        return (1.0 + gamma) * observation + beta


class _GaitMemory(_Memory):
    """Same LSTM memory, but the gait id modulates its input first (used by the actor)."""

    def __init__(self) -> None:
        super().__init__(input_size=73, hidden_size=512)
        self.gait_emb = nn.Embedding(4, 2)  # in the checkpoint, unused
        self.film_in = _FiLM()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        gait_id = torch.round(observation[..., -1]).long()
        # gait_id -> gait_code: 0 trot [0,0], 1 pronk [0,1], 2 pace [1,0], 3 bound [1,1]
        gait_code = torch.stack([(gait_id >> 1) & 1, gait_id & 1], dim=-1).float()
        return super().forward(self.film_in(observation, gait_code))


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
        self.memory_a = _GaitMemory()
        self.memory_c = _GaitMemory()

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.memory_a(observation))


# --- Gait clock: tells the model where each foot is in its walking cycle ---

class _GaitClock:
    TROT, PRONK, PACE, BOUND = 0, 1, 2, 3

    def __init__(self, step_dt: float) -> None:
        self.step_dt = step_dt
        self._t_global = 0.0

    def step(self, gait_frequency: float, gait_phase: float, gait_id: float) -> list[float]:
        self._t_global = (self._t_global + self.step_dt * gait_frequency) % 1.0
        if gait_id == self.PRONK:
            theta1, theta2, theta3 = 0.0, 0.0, 0.0
        elif gait_id == self.TROT:
            theta1, theta2, theta3 = gait_phase, 0.0, 0.0
        elif gait_id == self.PACE:
            theta1, theta2, theta3 = 0.0, gait_phase, 0.0
        elif gait_id == self.BOUND:
            theta1, theta2, theta3 = 0.0, 0.0, gait_phase

        t_foot = [
            (self._t_global + theta1 + theta2 + theta3) % 1.0,
            (self._t_global + theta2) % 1.0,
            (self._t_global + theta3) % 1.0,
            (self._t_global + theta1) % 1.0,
        ]
        return [math.sin(2.0 * math.pi * t) for t in t_foot]

# --- Policy: sensor readings + joystick commands -> joint targets via the trained model ---

class GaitwaysPolicy:
    """Loads the model once, then call `.step(...)` every control tick."""

    def __init__(self, model_path: str | Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor, self.estimator = self._load(Path(model_path), self.device)

        self.clock = _GaitClock(step_dt=1.0 / 50.0)
        self._action_t = [0.0] * NUM_JOINTS
        self._action_t_minus_1 = [0.0] * NUM_JOINTS

    # -- public API ----------------------------------------------------

    def step(
        self,
        base_angular_velocity: Sequence[float],   # 3 values, rad/s
        projected_gravity: Sequence[float],       # 3 values, unit vector
        current_joint_angles: Sequence[float],    # 12 values, rad
        current_joint_velocities: Sequence[float],  # 12 values, rad/s
        commands: Sequence[float],                # 12 values -- see docstring
    ) -> list[float]:
        """Run one control step. Returns 12 target joint angles (rad), in
        motor order.

        commands[6], [7] and [11] (gait frequency, phase, id) also drive the
        gait clock; all 12 are scaled by COMMANDS_SCALE into the observation.
        """

        # Joint position/velocity are re-ordered from motor order into the
        # policy's order (same mapping used for the output action), and
        # compared against each joint's default.
        joint_pos_error = [current_joint_angles[m] - DEFAULT_JOINT_ANGLES[m] for m in POLICY_TO_MOTOR]
        joint_vel_scaled = [0.05 * current_joint_velocities[m] for m in POLICY_TO_MOTOR]

        commands_scaled = [c * s for c, s in zip(commands, COMMANDS_SCALE)]

        # The gait clock is driven by specific slots inside `commands`,
        # not by fixed config values -- this matches gaitways_deploy's
        # GaitCommand message layout.
        clock_inputs = self.clock.step(
            gait_frequency=commands[6],
            gait_phase=commands[7],
            gait_id=commands[11],
        )

        observation = (
            [0.0, 0.0, 0.0]                       # placeholder for base_lin_vel
            + base_angular_velocity
            + projected_gravity
            + joint_pos_error
            + joint_vel_scaled
            + self._action_t
            + self._action_t_minus_1
            + clock_inputs
            + commands_scaled
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

        raw_action = raw_action.clamp(-10.0, 10.0).tolist()

        target_joint_angles = self._action_to_target_joint_angles(raw_action)

        self._action_t_minus_1 = self._action_t
        self._action_t = raw_action

        return target_joint_angles

    def build_lowcmd(self, motor_state, imu_state, commands: Sequence[float]) -> LowCmd:
        """Takes LowState's `motor_state`/`imu_state` fields plus a 12-value
        commands vector, runs the model, and returns a ready-to-publish
        LowCmd (CRC included)."""

        current_joint_angles = [motor_state[i].q for i in range(NUM_JOINTS)]
        current_joint_velocities = [motor_state[i].dq for i in range(NUM_JOINTS)]

        target_joint_angles = self.step(
            base_angular_velocity=list(imu_state.gyroscope),
            projected_gravity=_rpy_to_projected_gravity(imu_state.rpy),
            current_joint_angles=current_joint_angles,
            current_joint_velocities=current_joint_velocities,
            commands=commands,
        )

        return make_lowcmd(target_joint_angles)

    # -- internals -------------------------------------------------------

    def _action_to_target_joint_angles(self, raw_action: list[float]) -> list[float]:
        target_joint_angles = list(DEFAULT_JOINT_ANGLES)
        for policy_idx, motor_idx in enumerate(POLICY_TO_MOTOR):
            angle = DEFAULT_JOINT_ANGLES[motor_idx] + DEFAULT_ACTION_SCALE * raw_action[policy_idx]
            angle = max(HARD_LOWER_LIMITS[motor_idx], min(HARD_UPPER_LIMITS[motor_idx], angle))
            target_joint_angles[motor_idx] = angle
        return target_joint_angles

    @staticmethod
    def _load(model_path: Path, device: torch.device) -> tuple[_Actor, _Estimator]:
        checkpoint = torch.load(model_path.as_posix(), map_location=device)
        actor = _Actor()
        estimator = _Estimator()
        actor.load_state_dict(checkpoint["model_state_dict"])
        estimator.load_state_dict(checkpoint["estimator_state_dict"])
        actor.to(device).eval()
        estimator.to(device).eval()
        return actor, estimator


def make_lowcmd(target_joint_angles: Sequence[float]) -> LowCmd:
    """12 joint targets (motor order) -> ready-to-publish LowCmd."""
    cmd = LowCmd()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    for motor_idx in range(NUM_JOINTS):
        motor = cmd.motor_cmd[motor_idx]
        motor.mode = 0x01
        motor.q = target_joint_angles[motor_idx]
        motor.dq = 0.0
        motor.kp = MOTOR_KP
        motor.kd = MOTOR_KD
        motor.tau = 0.0
    _fill_crc(cmd)
    return cmd


# --- LowCmd CRC: fills in the CRC-32 checksum ---

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
