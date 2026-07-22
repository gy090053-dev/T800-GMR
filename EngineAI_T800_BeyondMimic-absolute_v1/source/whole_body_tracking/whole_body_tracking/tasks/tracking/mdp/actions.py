from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReferenceJointPositionAction(JointPositionAction):
    """Drive joints with q_target = raw_action * scale + q_ref(t)."""

    cfg: ReferenceJointPositionActionCfg

    def __init__(self, cfg: ReferenceJointPositionActionCfg, env: ManagerBasedEnv):
        cfg.use_default_offset = False
        super().__init__(cfg, env)
        self._offset = torch.zeros_like(self._raw_actions)

    def sync_reference_offset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Write current motion joint targets into the action offset buffer."""
        command = self._env.command_manager.get_term(self.cfg.command_name)
        ref_joint_pos = command.joint_pos
        if ref_joint_pos.shape[1] != self.action_dim:
            raise RuntimeError(
                f"Motion reference joints ({ref_joint_pos.shape[1]}) != action dim ({self.action_dim}). "
                "Align action joint_names with commands.motion.motion_joint_names."
            )
        if env_ids is None:
            env_ids = slice(None)
        self._offset[env_ids] = ref_joint_pos[env_ids]

    def process_actions(self, actions: torch.Tensor):
        if self.cfg.zero_policy_action:
            actions = torch.zeros_like(actions)
        self._raw_actions[:] = actions
        self.sync_reference_offset()
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        self.sync_reference_offset()
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with a time-varying reference offset from the motion command."""

    class_type: type[ActionTerm] = ReferenceJointPositionAction

    command_name: str = MISSING
    zero_policy_action: bool = False
