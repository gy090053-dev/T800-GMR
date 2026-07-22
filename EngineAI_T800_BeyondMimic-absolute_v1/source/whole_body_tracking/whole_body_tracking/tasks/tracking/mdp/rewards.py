from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _final_frame_gate(command: MotionCommand) -> torch.Tensor:  # for RESUME
    return command.is_holding_final_frame.to(torch.float32)  # for RESUME


def _terminal_quiet_gate(command: MotionCommand) -> torch.Tensor:  # This is altered for ending hold
    return (command.is_holding_final_frame & (command.phase_time_after_end < command.terminal_quiet_time_s)).to(  # This is altered for ending hold
        torch.float32  # This is altered for ending hold
    )  # This is altered for ending hold


def _terminal_recovery_gate(command: MotionCommand) -> torch.Tensor:  # This is altered for ending hold
    return (command.is_holding_final_frame & (command.phase_time_after_end >= command.terminal_quiet_time_s)).to(  # This is altered for ending hold
        torch.float32  # This is altered for ending hold
    )  # This is altered for ending hold


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    command.refresh_body_relative_w()
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    command.refresh_body_relative_w()
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def final_joint_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:  # for RESUME
    command: MotionCommand = env.command_manager.get_term(command_name)  # for RESUME
    gate = _final_frame_gate(command)  # for RESUME
    error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)  # for RESUME
    return gate * torch.exp(-error / std**2)  # for RESUME


def final_weighted_joint_position_error_exp(  # This is altered for ending hold
    env: ManagerBasedRLEnv, command_name: str, std: float, joint_weights: dict[str, float]  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    gate = _final_frame_gate(command)  # This is altered for ending hold
    weights = torch.ones(command.robot_joint_pos.shape[1], device=command.robot_joint_pos.device)  # This is altered for ending hold
    joint_names = command.cfg.motion_joint_names or command.robot.joint_names  # This is altered for ending hold
    for joint_id, joint_name in enumerate(joint_names):  # This is altered for ending hold
        weights[joint_id] = joint_weights.get(joint_name, 1.0)  # This is altered for ending hold
    error = torch.sum(weights.unsqueeze(0) * torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)  # This is altered for ending hold
    return gate * torch.exp(-error / std**2)  # This is altered for ending hold


def final_joint_position_deadband_error_exp(  # This is altered for ending hold
    env: ManagerBasedRLEnv,  # This is altered for ending hold
    command_name: str,  # This is altered for ending hold
    std: float,  # This is altered for ending hold
    target_joint_pos: dict[str, float],  # This is altered for ending hold
    tolerance: float | dict[str, float],  # This is altered for ending hold
    joint_weights: dict[str, float],  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    gate = _final_frame_gate(command)  # This is altered for ending hold
    joint_names = command.cfg.motion_joint_names or command.robot.joint_names  # This is altered for ending hold
    target = torch.empty(command.robot_joint_pos.shape[1], device=command.robot_joint_pos.device)  # This is altered for ending hold
    tolerances = torch.empty_like(target)  # This is altered for ending hold
    weights = torch.ones_like(target)  # This is altered for ending hold
    for joint_id, joint_name in enumerate(joint_names):  # This is altered for ending hold
        target[joint_id] = target_joint_pos[joint_name]  # This is altered for ending hold
        if isinstance(tolerance, dict):  # This is altered for ending hold
            tolerances[joint_id] = tolerance.get(joint_name, 0.0)  # This is altered for ending hold
        else:  # This is altered for ending hold
            tolerances[joint_id] = tolerance  # This is altered for ending hold
        weights[joint_id] = joint_weights.get(joint_name, 1.0)  # This is altered for ending hold
    error_per_joint = torch.relu(torch.abs(command.robot_joint_pos - target.unsqueeze(0)) - tolerances.unsqueeze(0))  # This is altered for ending hold
    error = torch.sum(weights.unsqueeze(0) * torch.square(error_per_joint), dim=-1)  # This is altered for ending hold
    return gate * torch.exp(-error / std**2)  # This is altered for ending hold


def final_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:  # for RESUME
    command: MotionCommand = env.command_manager.get_term(command_name)  # for RESUME
    gate = _final_frame_gate(command)  # for RESUME
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2  # for RESUME
    return gate * torch.exp(-error / std**2)  # for RESUME


def final_base_velocity_l2(  # for RESUME
    env: ManagerBasedRLEnv, command_name: str, lin_weight: float = 1.0, ang_weight: float = 0.25  # for RESUME
) -> torch.Tensor:  # for RESUME
    command: MotionCommand = env.command_manager.get_term(command_name)  # for RESUME
    gate = _final_frame_gate(command)  # for RESUME
    lin_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)  # for RESUME
    ang_error = torch.sum(torch.square(command.robot_anchor_ang_vel_w), dim=-1)  # for RESUME
    return gate * (lin_weight * lin_error + ang_weight * ang_error)  # for RESUME


def final_quiet_base_velocity_l2(  # This is altered for ending hold
    env: ManagerBasedRLEnv, command_name: str, lin_weight: float = 1.0, ang_weight: float = 0.25  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    gate = _terminal_quiet_gate(command)  # This is altered for ending hold
    lin_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)  # This is altered for ending hold
    ang_error = torch.sum(torch.square(command.robot_anchor_ang_vel_w), dim=-1)  # This is altered for ending hold
    return gate * (lin_weight * lin_error + ang_weight * ang_error)  # This is altered for ending hold


def final_recovery_base_velocity_l2(  # This is altered for ending hold
    env: ManagerBasedRLEnv, command_name: str, lin_weight: float = 1.0, ang_weight: float = 0.25  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    gate = _terminal_recovery_gate(command)  # This is altered for ending hold
    lin_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)  # This is altered for ending hold
    ang_error = torch.sum(torch.square(command.robot_anchor_ang_vel_w), dim=-1)  # This is altered for ending hold
    return gate * (lin_weight * lin_error + ang_weight * ang_error)  # This is altered for ending hold


def final_feet_slip_l2(  # for RESUME
    env: ManagerBasedRLEnv,  # for RESUME
    command_name: str,  # for RESUME
    sensor_cfg: SceneEntityCfg,  # for RESUME
    body_names: list[str],  # for RESUME
    contact_threshold: float = 0.0,  # for RESUME
) -> torch.Tensor:  # for RESUME
    command: MotionCommand = env.command_manager.get_term(command_name)  # for RESUME
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # for RESUME
    gate = _final_frame_gate(command)  # for RESUME
    body_indexes = _get_body_indexes(command, body_names)  # for RESUME
    in_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] > contact_threshold  # for RESUME
    foot_vel_xy = command.robot_body_lin_vel_w[:, body_indexes, :2]  # for RESUME
    slip = torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=(1, 2))  # for RESUME
    return gate * slip  # for RESUME


def final_quiet_feet_slip_l2(  # This is altered for ending hold
    env: ManagerBasedRLEnv,  # This is altered for ending hold
    command_name: str,  # This is altered for ending hold
    sensor_cfg: SceneEntityCfg,  # This is altered for ending hold
    body_names: list[str],  # This is altered for ending hold
    contact_threshold: float = 0.0,  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # This is altered for ending hold
    gate = _terminal_quiet_gate(command)  # This is altered for ending hold
    body_indexes = _get_body_indexes(command, body_names)  # This is altered for ending hold
    in_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] > contact_threshold  # This is altered for ending hold
    foot_vel_xy = command.robot_body_lin_vel_w[:, body_indexes, :2]  # This is altered for ending hold
    slip = torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=(1, 2))  # This is altered for ending hold
    return gate * slip  # This is altered for ending hold


def final_recovery_feet_slip_l2(  # This is altered for ending hold
    env: ManagerBasedRLEnv,  # This is altered for ending hold
    command_name: str,  # This is altered for ending hold
    sensor_cfg: SceneEntityCfg,  # This is altered for ending hold
    body_names: list[str],  # This is altered for ending hold
    contact_threshold: float = 0.0,  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # This is altered for ending hold
    gate = _terminal_recovery_gate(command)  # This is altered for ending hold
    body_indexes = _get_body_indexes(command, body_names)  # This is altered for ending hold
    in_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] > contact_threshold  # This is altered for ending hold
    foot_vel_xy = command.robot_body_lin_vel_w[:, body_indexes, :2]  # This is altered for ending hold
    slip = torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=(1, 2))  # This is altered for ending hold
    return gate * slip  # This is altered for ending hold


def final_joint_velocity_l2(  # This is altered for ending hold
    env: ManagerBasedRLEnv, command_name: str, joint_names: list[str] | None = None  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    gate = _final_frame_gate(command)  # This is altered for ending hold
    selected_joint_names = command.cfg.motion_joint_names or command.robot.joint_names  # This is altered for ending hold
    joint_indexes = [  # This is altered for ending hold
        i for i, name in enumerate(selected_joint_names) if (joint_names is None) or (name in joint_names)  # This is altered for ending hold
    ]  # This is altered for ending hold
    if len(joint_indexes) == 0:  # This is altered for ending hold
        return torch.zeros(command.robot_joint_vel.shape[0], device=command.robot_joint_vel.device)  # This is altered for ending hold
    return gate * torch.sum(torch.square(command.robot_joint_vel[:, joint_indexes]), dim=-1)  # This is altered for ending hold


def final_quiet_joint_velocity_l2(  # This is altered for ending hold
    env: ManagerBasedRLEnv, command_name: str, joint_names: list[str] | None = None  # This is altered for ending hold
) -> torch.Tensor:  # This is altered for ending hold
    command: MotionCommand = env.command_manager.get_term(command_name)  # This is altered for ending hold
    gate = _terminal_quiet_gate(command)  # This is altered for ending hold
    selected_joint_names = command.cfg.motion_joint_names or command.robot.joint_names  # This is altered for ending hold
    joint_indexes = [  # This is altered for ending hold
        i for i, name in enumerate(selected_joint_names) if (joint_names is None) or (name in joint_names)  # This is altered for ending hold
    ]  # This is altered for ending hold
    if len(joint_indexes) == 0:  # This is altered for ending hold
        return torch.zeros(command.robot_joint_vel.shape[0], device=command.robot_joint_vel.device)  # This is altered for ending hold
    return gate * torch.sum(torch.square(command.robot_joint_vel[:, joint_indexes]), dim=-1)  # This is altered for ending hold


def final_contact_stability(  # for RESUME
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, contact_threshold: float = 0.0  # for RESUME
) -> torch.Tensor:  # for RESUME
    command: MotionCommand = env.command_manager.get_term(command_name)  # for RESUME
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # for RESUME
    gate = _final_frame_gate(command)  # for RESUME
    in_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] > contact_threshold  # for RESUME
    return gate * in_contact.to(torch.float32).mean(dim=-1)  # for RESUME


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
