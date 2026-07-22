from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self._motion_joint_ids: torch.Tensor | None = None
        if self.cfg.motion_joint_names is not None:
            motion_joint_ids, _ = self.robot.find_joints(self.cfg.motion_joint_names, preserve_order=True)
            self._motion_joint_ids = torch.tensor(motion_joint_ids, dtype=torch.long, device=self.device)

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        num_motion_joints = self.motion.joint_pos.shape[1]
        if self._motion_joint_ids is not None:
            if len(self._motion_joint_ids) != num_motion_joints:
                raise ValueError(
                    f"Motion joint count ({num_motion_joints}) does not match motion_joint_names "
                    f"({len(self._motion_joint_ids)})."
                )
        elif num_motion_joints != self.robot.num_joints:
            raise ValueError(
                f"Motion joint count ({num_motion_joints}) does not match robot joint count ({self.robot.num_joints}). "
                "Provide motion_joint_names to map legacy motion joints onto the robot."
            )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        num_logged_joints = len(cfg.motion_joint_names) if cfg.motion_joint_names is not None else num_motion_joints
        self._reset_motion_joint_pos = torch.zeros(self.num_envs, num_logged_joints, device=self.device)
        self._reset_robot_joint_pos = torch.zeros(self.num_envs, num_logged_joints, device=self.device)
        num_tracked_bodies = len(cfg.body_names)
        self._reset_body_pos_relative_w = torch.zeros(self.num_envs, num_tracked_bodies, 3, device=self.device)
        self._reset_robot_body_pos_w = torch.zeros(self.num_envs, num_tracked_bodies, 3, device=self.device)
        self._step1_motion_joint_pos = torch.zeros(self.num_envs, num_logged_joints, device=self.device)
        self._step1_body_pos_relative_w = torch.zeros(self.num_envs, num_tracked_bodies, 3, device=self.device)
        self._pending_step1_snapshot = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_holding_final_frame = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # for RESUME
        self._hold_steps_after_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # for RESUME
        self._terminal_quiet_time_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)  # This is altered for ending hold

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        joint_vel = self.motion.joint_vel[self.time_steps].clone()  # for RESUME
        joint_vel[self._is_holding_final_frame] = 0.0  # for RESUME
        return joint_vel  # for RESUME

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        body_lin_vel_w = self.motion.body_lin_vel_w[self.time_steps].clone()  # for RESUME
        body_lin_vel_w[self._is_holding_final_frame] = 0.0  # for RESUME
        return body_lin_vel_w  # for RESUME

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        body_ang_vel_w = self.motion.body_ang_vel_w[self.time_steps].clone()  # for RESUME
        body_ang_vel_w[self._is_holding_final_frame] = 0.0  # for RESUME
        return body_ang_vel_w  # for RESUME

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        if self._motion_joint_ids is not None:
            return self.robot.data.joint_pos[:, self._motion_joint_ids]
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        if self._motion_joint_ids is not None:
            return self.robot.data.joint_vel[:, self._motion_joint_ids]
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property  # for RESUME
    def is_holding_final_frame(self) -> torch.Tensor:  # for RESUME
        return self._is_holding_final_frame  # for RESUME

    @property  # for RESUME
    def phase_time_after_end(self) -> torch.Tensor:  # for RESUME
        return self._hold_steps_after_end.to(torch.float32) * self._env.step_dt  # for RESUME

    @property  # This is altered for ending hold
    def terminal_quiet_time_s(self) -> torch.Tensor:  # This is altered for ending hold
        return self._terminal_quiet_time_s  # This is altered for ending hold

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.sampling_mode == "adaptive":  # for RESUME
            self._adaptive_sampling(env_ids)  # for RESUME
        elif self.cfg.sampling_mode == "sequential":  # for RESUME
            self.time_steps[env_ids] = 0  # for RESUME
            self.metrics["sampling_entropy"][:] = 1.0  # for RESUME
            self.metrics["sampling_top1_prob"][:] = 0.0  # for RESUME
            self.metrics["sampling_top1_bin"][:] = 0.0  # for RESUME
        else:  # for RESUME
            raise ValueError(f"Unsupported motion sampling_mode: {self.cfg.sampling_mode}")  # for RESUME
        self._is_holding_final_frame[env_ids] = False  # for RESUME
        self._hold_steps_after_end[env_ids] = 0  # for RESUME
        self._terminal_quiet_time_s[env_ids] = 0.0  # This is altered for ending hold

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        if self._motion_joint_ids is not None:
            joint_pos[env_ids] = torch.clip(
                joint_pos[env_ids],
                soft_joint_pos_limits[:, self._motion_joint_ids, 0],
                soft_joint_pos_limits[:, self._motion_joint_ids, 1],
            )
            sim_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            sim_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
            sim_joint_pos[:, self._motion_joint_ids] = joint_pos[env_ids]
            sim_joint_vel[:, self._motion_joint_ids] = joint_vel[env_ids]
            self._reset_motion_joint_pos[env_ids] = joint_pos[env_ids]
            self.robot.write_joint_state_to_sim(sim_joint_pos, sim_joint_vel, env_ids=env_ids)
        else:
            joint_pos[env_ids] = torch.clip(
                joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
            )
            self._reset_motion_joint_pos[env_ids] = joint_pos[env_ids]
            self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        self.refresh_body_relative_w()
        self._reset_body_pos_relative_w[env_ids] = self.body_pos_relative_w[env_ids]
        self._reset_robot_joint_pos[env_ids] = self.robot_joint_pos[env_ids]
        self._reset_robot_body_pos_w[env_ids] = self.robot_body_pos_w[env_ids]
        self._pending_step1_snapshot[env_ids] = True
        self._sync_action_reference_offset(env_ids)

    def _sync_action_reference_offset(self, env_ids: Sequence[int]) -> None:
        """Keep action offset aligned with the motion frame written at reset/resample."""
        if len(env_ids) == 0:
            return
        action_term = self._env.action_manager.get_term("joint_pos")
        if hasattr(action_term, "sync_reference_offset"):
            env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
            action_term.sync_reference_offset(env_ids_tensor)

    def refresh_body_relative_w(self) -> None:
        """Recompute goal body poses from the current robot anchor and motion frame."""
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _capture_step1_reference_snapshot(self, env_mask: torch.Tensor) -> None:
        """Snapshot motion reference after the first command step following reset."""
        env_ids = env_mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        self._step1_motion_joint_pos[env_ids] = self.joint_pos[env_ids]
        self._step1_body_pos_relative_w[env_ids] = self.body_pos_relative_w[env_ids]
        self._pending_step1_snapshot[env_ids] = False

    def _update_command(self):
        self.time_steps += 1
        end_env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]  # for RESUME
        step1_candidates = self._pending_step1_snapshot.clone()
        if len(end_env_ids) > 0:  # for RESUME
            step1_candidates[end_env_ids] = False  # for RESUME

        if self.cfg.end_behavior == "resample":  # for RESUME
            self._resample_command(end_env_ids)  # for RESUME
        elif self.cfg.end_behavior == "hold":  # for RESUME
            newly_holding_env_ids = end_env_ids[~self._is_holding_final_frame[end_env_ids]]  # This is altered for ending hold
            if len(newly_holding_env_ids) > 0:  # This is altered for ending hold
                quiet_range = torch.tensor(self.cfg.terminal_quiet_time_range_s, device=self.device)  # This is altered for ending hold
                self._terminal_quiet_time_s[newly_holding_env_ids] = sample_uniform(  # This is altered for ending hold
                    quiet_range[0], quiet_range[1], (len(newly_holding_env_ids),), device=self.device  # This is altered for ending hold
                )  # This is altered for ending hold
            self._is_holding_final_frame[end_env_ids] = True  # for RESUME
            self._hold_steps_after_end[self._is_holding_final_frame] += 1  # for RESUME
            self.time_steps.clamp_(max=self.motion.time_step_total - 1)  # for RESUME
            self._sync_action_reference_offset(end_env_ids)  # for RESUME
        else:  # for RESUME
            raise ValueError(f"Unsupported motion end_behavior: {self.cfg.end_behavior}")  # for RESUME

        self.refresh_body_relative_w()
        self._capture_step1_reference_snapshot(step1_candidates)

        if self.cfg.sampling_mode == "adaptive":  # for RESUME
            self.bin_failed_count = (  # for RESUME
                self.cfg.adaptive_alpha * self._current_bin_failed  # for RESUME
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count  # for RESUME
            )  # for RESUME
            self._current_bin_failed.zero_()  # for RESUME

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    motion_joint_names: list[str] | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    sampling_mode: str = "adaptive"  # for RESUME
    end_behavior: str = "resample"  # for RESUME
    terminal_quiet_time_range_s: tuple[float, float] = (0.0, 0.0)  # This is altered for ending hold

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
