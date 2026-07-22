"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --input_file /path/to/motion.npz --robot t800
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--input_file", type=str, default=None, help="Path to a local .npz motion file.")
parser.add_argument("--robot", type=str, default="t800", choices=["t800"], help="Robot type to use.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from pathlib import Path

# Allow running from repo root without a prior editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_WBT_SOURCE = _REPO_ROOT / "source" / "whole_body_tracking"
if _WBT_SOURCE.is_dir() and str(_WBT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_WBT_SOURCE))

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.t800 import T800_CFG, T800_MOTION_JOINT_NAMES
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

ROBOT_CFGS = {
    "t800": T800_CFG,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (will be overridden in main based on --robot)
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _motion_root_body_column_npz(npz_body_columns: int) -> int:
    """Column index into ``body_pos_w`` for the floating base (``LINK_BASE``).

    ``repaired_npy_to_npz`` and ``npy_to_npz`` both store the root at column 0 in
    the npz body layout. Do not use ``robot.find_bodies`` here — those are simulator
    body indices, not npz channel indices.
    """
    if npz_body_columns < 1:
        raise ValueError(f"NPZ body axis must be positive, got {npz_body_columns}")
    return 0


def _resolve_robot_joint_indexes(robot: Articulation, num_motion_joints: int, robot_name: str) -> list[int]:
    """Map npz joint columns (``T800_MOTION_JOINT_NAMES`` order) onto PhysX joint indices."""
    if robot_name != "t800":
        raise ValueError(f"Unsupported robot for motion joint mapping: {robot_name}")

    robot_joint_indexes = robot.find_joints(T800_MOTION_JOINT_NAMES, preserve_order=True)[0]
    if len(robot_joint_indexes) != num_motion_joints:
        raise ValueError(
            f"Motion joint count ({num_motion_joints}) does not match "
            f"T800_MOTION_JOINT_NAMES ({len(T800_MOTION_JOINT_NAMES)}). "
            "Convert npz with scripts/repaired_npy_to_npz.py or ensure joint columns "
            "follow T800_MOTION_JOINT_NAMES."
        )
    print(f"[replay_npz] Mapping {num_motion_joints} motion joints via T800_MOTION_JOINT_NAMES.")
    return robot_joint_indexes


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.input_file is not None:
        motion_file = args_cli.input_file
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Either --input_file or --registry_name must be provided.")

    probe = np.load(motion_file)
    root_col = _motion_root_body_column_npz(int(probe["body_pos_w"].shape[1]))

    motion = MotionLoader(
        motion_file,
        torch.tensor([root_col], dtype=torch.long, device=sim.device),
        sim.device,
    )
    robot_joint_indexes = _resolve_robot_joint_indexes(robot, motion.joint_pos.shape[1], args_cli.robot)
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps]
        joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
