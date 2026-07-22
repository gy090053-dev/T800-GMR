from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

_DEAD_LOG_FILENAME = "dead_log.txt"


def _format_bodies_xyz(body_pos_w: torch.Tensor) -> str:
    """Format supervised link positions as [x,y,z] tuples in body_names order."""
    return ", ".join(
        f"[{pos[0].item():.6f},{pos[1].item():.6f},{pos[2].item():.6f}]" for pos in body_pos_w
    )


def _format_log_line(joint_pos: torch.Tensor, body_pos_w: torch.Tensor) -> str:
    """Format one log line: joint angles followed by supervised link xyz."""
    joints_str = " ".join(f"{value:.6f}" for value in joint_pos.tolist())
    bodies_str = _format_bodies_xyz(body_pos_w)
    return f"{joints_str} {bodies_str}"


def log_ee_body_pos_dead(env: ManagerBasedRLEnv, command_name: str, triggered_env_ids: torch.Tensor) -> None:
    """Append reset/death reference and actual snapshots when ee_body_pos terminates envs.

    Each record contains four lines:
    1. reference joints/bodies at step 1 after reset
    2. actual robot joints/bodies at reset
    3. reference joints/bodies at termination
    4. actual robot joints/bodies at termination
    """
    log_dir = getattr(env.cfg, "log_dir", None)
    if not log_dir:
        return

    log_path = os.path.join(log_dir, _DEAD_LOG_FILENAME)
    command: MotionCommand = env.command_manager.get_term(command_name)

    lines: list[str] = []
    for env_id in triggered_env_ids.tolist():
        lines.append(_format_log_line(command._step1_motion_joint_pos[env_id], command._step1_body_pos_relative_w[env_id]))
        lines.append(_format_log_line(command._reset_robot_joint_pos[env_id], command._reset_robot_body_pos_w[env_id]))
        lines.append(_format_log_line(command.joint_pos[env_id], command.body_pos_relative_w[env_id]))
        lines.append(_format_log_line(command.robot_joint_pos[env_id], command.robot_body_pos_w[env_id]))
        lines.append("")

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))
        if lines:
            log_file.write("\n")
