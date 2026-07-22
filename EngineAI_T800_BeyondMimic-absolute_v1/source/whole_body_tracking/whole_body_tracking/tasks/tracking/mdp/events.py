from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def push_robot_after_final_frame(  # This is altered for ending hold
    env: ManagerBasedEnv,  # This is altered for ending hold
    env_ids: torch.Tensor | None,  # This is altered for ending hold
    command_name: str,  # This is altered for ending hold
    velocity_range: dict[str, tuple[float, float]],  # This is altered for ending hold
    push_probability: float = 1.0,  # This is altered for ending hold
    hold_grace_time_s: float = 0.0,  # This is altered for ending hold
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # This is altered for ending hold
):  # This is altered for ending hold
    asset: Articulation = env.scene[asset_cfg.name]  # This is altered for ending hold
    command = env.command_manager.get_term(command_name)  # This is altered for ending hold
    if env_ids is None:  # This is altered for ending hold
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)  # This is altered for ending hold
    else:  # This is altered for ending hold
        env_ids = env_ids.to(asset.device)  # This is altered for ending hold
    holding_env_ids = env_ids[command.is_holding_final_frame[env_ids]]  # This is altered for ending hold
    if len(holding_env_ids) == 0:  # This is altered for ending hold
        return  # This is altered for ending hold
    grace_time_s = torch.maximum(  # This is altered for ending hold
        command.terminal_quiet_time_s[holding_env_ids],  # This is altered for ending hold
        torch.full((len(holding_env_ids),), hold_grace_time_s, device=asset.device),  # This is altered for ending hold
    )  # This is altered for ending hold
    grace_mask = command.phase_time_after_end[holding_env_ids] >= grace_time_s  # This is altered for ending hold
    holding_env_ids = holding_env_ids[grace_mask]  # This is altered for ending hold
    if len(holding_env_ids) == 0:  # This is altered for ending hold
        return  # This is altered for ending hold
    if push_probability <= 0.0:  # This is altered for ending hold
        return  # This is altered for ending hold
    if push_probability < 1.0:  # This is altered for ending hold
        push_mask = torch.rand(len(holding_env_ids), device=asset.device) < push_probability  # This is altered for ending hold
        holding_env_ids = holding_env_ids[push_mask]  # This is altered for ending hold
        if len(holding_env_ids) == 0:  # This is altered for ending hold
            return  # This is altered for ending hold
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  # This is altered for ending hold
    ranges = torch.tensor(range_list, device=asset.device)  # This is altered for ending hold
    random_velocity = math_utils.sample_uniform(  # This is altered for ending hold
        ranges[:, 0], ranges[:, 1], (len(holding_env_ids), 6), device=asset.device  # This is altered for ending hold
    )  # This is altered for ending hold
    root_velocity = asset.data.root_vel_w[holding_env_ids].clone()  # This is altered for ending hold
    root_velocity += random_velocity  # This is altered for ending hold
    asset.write_root_velocity_to_sim(root_velocity, env_ids=holding_env_ids)  # This is altered for ending hold


def push_robot_before_final_frame(  # This is altered for ending hold
    env: ManagerBasedEnv,  # This is altered for ending hold
    env_ids: torch.Tensor | None,  # This is altered for ending hold
    command_name: str,  # This is altered for ending hold
    velocity_range: dict[str, tuple[float, float]],  # This is altered for ending hold
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # This is altered for ending hold
):  # This is altered for ending hold
    asset: Articulation = env.scene[asset_cfg.name]  # This is altered for ending hold
    command = env.command_manager.get_term(command_name)  # This is altered for ending hold
    if env_ids is None:  # This is altered for ending hold
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)  # This is altered for ending hold
    else:  # This is altered for ending hold
        env_ids = env_ids.to(asset.device)  # This is altered for ending hold
    tracking_env_ids = env_ids[~command.is_holding_final_frame[env_ids]]  # This is altered for ending hold
    if len(tracking_env_ids) == 0:  # This is altered for ending hold
        return  # This is altered for ending hold
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  # This is altered for ending hold
    ranges = torch.tensor(range_list, device=asset.device)  # This is altered for ending hold
    random_velocity = math_utils.sample_uniform(  # This is altered for ending hold
        ranges[:, 0], ranges[:, 1], (len(tracking_env_ids), 6), device=asset.device  # This is altered for ending hold
    )  # This is altered for ending hold
    root_velocity = asset.data.root_vel_w[tracking_env_ids].clone()  # This is altered for ending hold
    root_velocity += random_velocity  # This is altered for ending hold
    asset.write_root_velocity_to_sim(root_velocity, env_ids=tracking_env_ids)  # This is altered for ending hold


def push_robot_after_final_frame_with_forward_bias(  # This is altered for ending hold
    env: ManagerBasedEnv,  # This is altered for ending hold
    env_ids: torch.Tensor | None,  # This is altered for ending hold
    command_name: str,  # This is altered for ending hold
    velocity_range: dict[str, tuple[float, float]],  # This is altered for ending hold
    forward_x_range: tuple[float, float],  # This is altered for ending hold
    forward_probability: float = 0.5,  # This is altered for ending hold
    push_probability: float = 1.0,  # This is altered for ending hold
    hold_grace_time_s: float = 0.0,  # This is altered for ending hold
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # This is altered for ending hold
):  # This is altered for ending hold
    asset: Articulation = env.scene[asset_cfg.name]  # This is altered for ending hold
    command = env.command_manager.get_term(command_name)  # This is altered for ending hold
    if env_ids is None:  # This is altered for ending hold
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)  # This is altered for ending hold
    else:  # This is altered for ending hold
        env_ids = env_ids.to(asset.device)  # This is altered for ending hold
    holding_env_ids = env_ids[command.is_holding_final_frame[env_ids]]  # This is altered for ending hold
    if len(holding_env_ids) == 0:  # This is altered for ending hold
        return  # This is altered for ending hold
    grace_mask = command.phase_time_after_end[holding_env_ids] >= hold_grace_time_s  # This is altered for ending hold
    holding_env_ids = holding_env_ids[grace_mask]  # This is altered for ending hold
    if len(holding_env_ids) == 0:  # This is altered for ending hold
        return  # This is altered for ending hold
    if push_probability <= 0.0:  # This is altered for ending hold
        return  # This is altered for ending hold
    if push_probability < 1.0:  # This is altered for ending hold
        push_mask = torch.rand(len(holding_env_ids), device=asset.device) < push_probability  # This is altered for ending hold
        holding_env_ids = holding_env_ids[push_mask]  # This is altered for ending hold
        if len(holding_env_ids) == 0:  # This is altered for ending hold
            return  # This is altered for ending hold
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  # This is altered for ending hold
    ranges = torch.tensor(range_list, device=asset.device)  # This is altered for ending hold
    random_velocity = math_utils.sample_uniform(  # This is altered for ending hold
        ranges[:, 0], ranges[:, 1], (len(holding_env_ids), 6), device=asset.device  # This is altered for ending hold
    )  # This is altered for ending hold
    forward_mask = torch.rand(len(holding_env_ids), device=asset.device) < forward_probability  # This is altered for ending hold
    if torch.any(forward_mask):  # This is altered for ending hold
        forward_range = torch.tensor(forward_x_range, device=asset.device)  # This is altered for ending hold
        random_velocity[forward_mask, 0] = math_utils.sample_uniform(  # This is altered for ending hold
            forward_range[0], forward_range[1], (int(forward_mask.sum().item()),), device=asset.device  # This is altered for ending hold
        )  # This is altered for ending hold
    root_velocity = asset.data.root_vel_w[holding_env_ids].clone()  # This is altered for ending hold
    root_velocity += random_velocity  # This is altered for ending hold
    asset.write_root_velocity_to_sim(root_velocity, env_ids=holding_env_ids)  # This is altered for ending hold
