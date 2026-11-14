from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_JOINT_SIGNS = {
    "hip_pitch_joint": 1.0,
    "hip_roll_joint": -1.0,
    "hip_yaw_joint": -1.0,
    "knee_joint": 1.0,
    "ankle_pitch_joint": 1.0,
    "ankle_roll_joint": -1.0,
    "shoulder_pitch_joint": 1.0,
    "shoulder_roll_joint": -1.0,
    "shoulder_yaw_joint": -1.0,
    "elbow_joint": 1.0,
    "wrist_roll_joint": -1.0,
    "wrist_pitch_joint": 1.0,
    "wrist_yaw_joint": -1.0,
    "waist_yaw_joint": -1.0,
    "waist_roll_joint": -1.0,
    "waist_pitch_joint": 1.0,
}

_VECTOR_SIGNS = {
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "velocity_commands": (1.0, -1.0, -1.0),
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


@lru_cache(maxsize=8)
def _joint_mirror_map(joint_names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    indices = {name: i for i, name in enumerate(joint_names)}
    permutation, signs = [], []
    for name in joint_names:
        if name.startswith("left_"):
            kind, counterpart = name[5:], "right_" + name[5:]
        elif name.startswith("right_"):
            kind, counterpart = name[6:], "left_" + name[6:]
        else:
            kind, counterpart = name, name
        if kind not in _JOINT_SIGNS or counterpart not in indices:
            raise ValueError(f"No G1 mirror rule or matching joint for '{name}'.")
        permutation.append(indices[counterpart])
        signs.append(_JOINT_SIGNS[kind])
    return tuple(permutation), tuple(signs)


def mirror_joints(x: torch.Tensor, joint_names: list[str] | tuple[str, ...]) -> torch.Tensor:
    """Mirror G1 joint values in their actual observation/action order (23 or 29 DoF)."""
    if x.shape[-1] != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint values, got {x.shape[-1]}.")
    permutation, signs = _joint_mirror_map(tuple(joint_names))
    sign = torch.tensor(signs, device=x.device, dtype=x.dtype)
    return x[..., list(permutation)] * sign


def _mirror_height_scan(x: torch.Tensor, env: ManagerBasedRLEnv, term_cfg) -> torch.Tensor:
    pattern = env.scene[term_cfg.params["sensor_cfg"].name].cfg.pattern_cfg
    nx = int(round(pattern.size[0] / pattern.resolution)) + 1
    ny = int(round(pattern.size[1] / pattern.resolution)) + 1
    return x.view(-1, ny, nx).flip(dims=[1]).reshape(x.shape)


def _mirror_foothold_scan(x: torch.Tensor, term_cfg) -> torch.Tensor:
    k = term_cfg.params["patch_size"]
    stride = 2 + k * k
    feet = [x[:, :stride], x[:, stride:]]
    out = torch.empty_like(x)
    for i, src in enumerate(reversed(feet)):
        o = i * stride
        out[:, o] = src[:, 0]
        out[:, o + 1] = -src[:, 1]
        out[:, o + 2 : o + stride] = src[:, 2:].view(-1, k, k).flip(dims=[2]).reshape(-1, k * k)
    return out


def _mirror_group(env: ManagerBasedRLEnv, obs: torch.Tensor, group: str) -> torch.Tensor:
    obs_manager = env.observation_manager
    group_cfg = getattr(env.cfg.observations, group)
    out = obs.clone()
    idx = 0
    for name, shape in zip(obs_manager.active_terms[group], obs_manager.group_obs_term_dim[group]):
        dim = math.prod(shape)
        seg = obs[:, idx : idx + dim]
        if name in _VECTOR_SIGNS:
            out[:, idx : idx + dim] = seg * torch.tensor(_VECTOR_SIGNS[name], device=obs.device)
        elif name in _JOINT_TERMS:
            out[:, idx : idx + dim] = mirror_joints(seg, env.scene["robot"].joint_names)
        elif name == "height_scan":
            out[:, idx : idx + dim] = _mirror_height_scan(seg, env, getattr(group_cfg, name))
        elif name == "foothold_scan":
            out[:, idx : idx + dim] = _mirror_foothold_scan(seg, getattr(group_cfg, name))
        else:
            raise ValueError(f"No mirror rule for observation term '{name}' in group '{group}'.")
        idx += dim
    return out


def g1_symmetry_augmentation(
    env, obs: TensorDict | None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    env = env.unwrapped
    actions_aug = (
        None if actions is None else torch.cat([actions, mirror_joints(actions, env.scene["robot"].joint_names)], dim=0)
    )
    if obs is None:
        return None, actions_aug
    batch = obs.batch_size[0]
    aug = {}
    for key in obs.keys():
        tensor = obs[key]
        mirrored = _mirror_group(env, tensor, key) if key in ("policy", "critic") else tensor
        aug[key] = torch.cat([tensor, mirrored], dim=0)
    return TensorDict(aug, batch_size=[2 * batch], device=obs.device), actions_aug
