"""CPU regression checks; these do not require Isaac Sim to be running."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

import fpm  # noqa: F401
from fpm.foot_pred.model import FootPredictor, NormalizedFootPredictor, load_foot_predictor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g1_symmetry", ROOT / "fpm/mdp/symmetry.py")
symmetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry)


def joint_names(dofs):
    bilateral = [
        "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
    ]
    waist = ["waist_yaw_joint"]
    if dofs == 29:
        bilateral += ["wrist_pitch", "wrist_yaw"]
        waist += ["waist_roll_joint", "waist_pitch_joint"]
    # Deliberately use an order different from Isaac's usual articulation order.
    return sorted([f"{side}_{kind}_joint" for side in ("left", "right") for kind in bilateral] + waist)


@pytest.mark.parametrize("dofs", [23, 29])
def test_joint_mirror_is_independent_of_order_and_is_an_involution(dofs):
    names = joint_names(dofs)
    x = torch.arange(2 * dofs, dtype=torch.float32).reshape(2, dofs)
    mirrored = symmetry.mirror_joints(x, names)
    torch.testing.assert_close(symmetry.mirror_joints(mirrored, names), x)
    order = torch.randperm(dofs, generator=torch.Generator().manual_seed(11))
    shuffled_names = [names[i] for i in order.tolist()]
    torch.testing.assert_close(symmetry.mirror_joints(x[:, order], shuffled_names), mirrored[:, order])
    # A pitch angle keeps its sign; a roll/yaw angle changes sign under reflection.
    for left, right, sign in [
        ("left_hip_pitch_joint", "right_hip_pitch_joint", 1),
        ("left_ankle_roll_joint", "right_ankle_roll_joint", -1),
        ("left_wrist_roll_joint", "right_wrist_roll_joint", -1),
    ]:
        torch.testing.assert_close(mirrored[:, names.index(left)], sign * x[:, names.index(right)])
    torch.testing.assert_close(mirrored[:, names.index("waist_yaw_joint")], -x[:, names.index("waist_yaw_joint")])


@pytest.mark.parametrize("names", [["left_knee_joint"], ["unknown_joint"]])
def test_incomplete_or_unknown_joint_mapping_fails(names):
    with pytest.raises(ValueError, match="mirror rule"):
        symmetry.mirror_joints(torch.zeros(1, len(names)), names)


def test_joint_dimension_mismatch_fails():
    with pytest.raises(ValueError, match="Expected 23"):
        symmetry.mirror_joints(torch.zeros(1, 29), joint_names(23))


@pytest.mark.parametrize("dofs", [23, 29])
def test_full_policy_and_critic_augmentation(dofs):
    names = joint_names(dofs)
    common = {
        "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
        "joint_pos": dofs, "joint_vel": dofs, "actions": dofs,
    }
    terms = {
        "policy": {**common, "foothold_scan": 102},
        "critic": {**common, "height_scan": 693, "foothold_scan": 102},
    }
    group_cfg = SimpleNamespace(
        foothold_scan=SimpleNamespace(params={"patch_size": 7}),
        height_scan=SimpleNamespace(params={"sensor_cfg": SimpleNamespace(name="height_scanner")}),
    )
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(joint_names=names),
            "height_scanner": SimpleNamespace(cfg=SimpleNamespace(
                pattern_cfg=SimpleNamespace(size=(1.6, 1.0), resolution=0.05))),
        },
        observation_manager=SimpleNamespace(
            active_terms={k: list(v) for k, v in terms.items()},
            group_obs_term_dim={k: [(dim,) for dim in v.values()] for k, v in terms.items()},
        ),
        cfg=SimpleNamespace(observations=SimpleNamespace(policy=group_cfg, critic=group_cfg)),
    )
    env.unwrapped = env
    obs = TensorDict({k: torch.randn(2, sum(v.values())) for k, v in terms.items()}, batch_size=[2])
    actions = torch.randn(2, dofs)
    augmented, actions_aug = symmetry.g1_symmetry_augmentation(env, obs, actions)
    assert augmented.batch_size == torch.Size([4])
    for group in terms:
        torch.testing.assert_close(augmented[group][:2], obs[group])
        torch.testing.assert_close(symmetry._mirror_group(env, augmented[group][2:], group), obs[group])
    torch.testing.assert_close(actions_aug[2:], symmetry.mirror_joints(actions, names))
    if dofs == 23:
        assert obs["policy"].shape[-1] == 183
        assert obs["critic"].shape[-1] == 876


def test_foothold_mirror_swaps_feet_and_reflects_lateral_patch_axis():
    scan = torch.arange(102, dtype=torch.float32).reshape(1, 102)
    cfg = SimpleNamespace(params={"patch_size": 7})
    out = symmetry._mirror_foothold_scan(scan, cfg)
    assert out[0, 0] == scan[0, 51]
    assert out[0, 1] == -scan[0, 52]
    torch.testing.assert_close(out[:, 2:51].reshape(7, 7), scan[:, 53:].reshape(7, 7).flip(1))


def test_fpm_checkpoint_accepts_58_values_and_rejects_70_value_model(tmp_path):
    for dim in (58, 70):
        model = NormalizedFootPredictor(FootPredictor(dim, 4), torch.zeros(dim), torch.ones(dim)).eval()
        path = str(tmp_path / f"foot_{dim}.pt")
        torch.jit.save(torch.jit.script(model), path)
        if dim == 58:
            loaded = load_foot_predictor(path, 58)
            assert loaded(torch.zeros(2, 58)).shape == (2, 4)
        else:
            with pytest.raises(ValueError, match="expects 70 state values.*requires 58"):
                load_foot_predictor(path, 58)


@pytest.mark.parametrize("suffix,env_class", [
    ("", "FpmG1_23EnvCfg"),
    ("-Play", "FpmG1_23EnvCfg_PLAY"),
    ("-Validate-Locomotion", "FpmG1_23LocomotionValidationEnvCfg"),
    ("-Validate-FootPred", "FpmG1_23FootPredValidationEnvCfg"),
    ("-Validate-Corridor", "FpmG1_23CorridorValidationEnvCfg"),
])
def test_tasks_register_separate_robot_and_runner(suffix, env_class):
    task = gym.spec(f"FPM-G1-23{suffix}-v0")
    assert task.kwargs["env_cfg_entry_point"] == f"fpm.env_cfg:{env_class}"
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":FpmG1_23PPORunnerCfg")
    assert gym.spec(f"FPM-G1{suffix}-v0").kwargs["rsl_rl_cfg_entry_point"].endswith(":FpmPPORunnerCfg")


def load_checkpoint_resolver():
    tree = ast.parse((ROOT / "scripts/play.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("latest_checkpoint", "resolve_checkpoint")]
    import glob
    import os
    import re
    namespace = dict(glob=glob, os=os, re=re, args_cli=SimpleNamespace(checkpoint=None))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "play.py", "exec"), namespace)
    return namespace


def test_g1_23_playback_never_falls_back_to_29_dof_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolver = load_checkpoint_resolver()
    demo = tmp_path / "demo"
    demo.mkdir()
    checkpoint = demo / "model_10000.pt"
    checkpoint.touch()
    resolver["DEMO_DIR"] = str(demo)
    with pytest.raises(FileNotFoundError, match="29-DoF"):
        resolver["resolve_checkpoint"]("g1_23_fpm")
    assert resolver["resolve_checkpoint"]("g1_fpm") == str(checkpoint)
    run = tmp_path / "logs/rsl_rl/g1_23_fpm/run"
    run.mkdir(parents=True)
    (run / "model_2.pt").touch()
    (run / "model_10.pt").touch()
    assert Path(resolver["resolve_checkpoint"]("g1_23_fpm")).name == "model_10.pt"
