import os

import numpy as np
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict

OBS_NAME_ALIAS = {
    "motion_command": "command",
    "joint_pos_rel": "joint_pos",
    "joint_vel_rel": "joint_vel",
    "last_action": "actions",
}


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    if isinstance(x, list):
        return [format_value(i) for i in x]
    if isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    return x


def _get_policy_action_term(env: ManagerBasedRLEnv):
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        if isinstance(term, JointPositionAction):
            return term
    raise RuntimeError("No JointPositionAction term found in action manager.")


def _ordered_joint_data(asset: Articulation, joint_ids: np.ndarray):
    default_joint_pos = asset.data.default_joint_pos_nominal.detach().cpu().numpy()
    joint_stiffness = asset.data.joint_stiffness[0].detach().cpu().numpy()
    joint_damping = asset.data.joint_damping[0].detach().cpu().numpy()
    joint_names = [asset.data.joint_names[int(i)] for i in joint_ids]
    return (
        joint_names,
        default_joint_pos[joint_ids].tolist(),
        joint_stiffness[joint_ids].tolist(),
        joint_damping[joint_ids].tolist(),
    )


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir: str) -> str:
    """Export deployment metadata consumed by engineai_robotics_native_sdk."""
    asset: Articulation = env.scene["robot"]
    action_term = _get_policy_action_term(env)
    if action_term._joint_ids == slice(None):
        joint_ids = np.arange(asset.num_joints, dtype=np.int64)
    else:
        joint_ids = np.asarray(action_term._joint_ids, dtype=np.int64)
    policy_joint_names, default_joint_pos, joint_stiffness, joint_damping = _ordered_joint_data(asset, joint_ids)

    cfg = {}
    cfg["body_names"] = asset.data.body_names
    cfg["joint_names"] = policy_joint_names
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    cfg["default_joint_pos"] = default_joint_pos
    cfg["joint_stiffness"] = joint_stiffness
    cfg["joint_damping"] = joint_damping

    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "joint_pos"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        term_cfg = term_cfg.to_dict()
        for key in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            if key in term_cfg:
                del term_cfg[key]
        if "joint_names" in term_cfg:
            term_cfg["joint_names"] = policy_joint_names
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = list(range(action_term.action_dim))
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    cfg["observations"] = {}
    for obs_name, obs_cfg in zip(obs_names, obs_cfgs):
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        term_cfg = term_cfg.to_dict()
        for key in ["func", "modifiers", "noise", "flatten_history_dim"]:
            if key in term_cfg:
                del term_cfg[key]
        cfg["observations"][OBS_NAME_ALIAS.get(obs_name, obs_name)] = term_cfg

    filename = os.path.join(log_dir, "params", "deploy.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    return filename
