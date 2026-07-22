"""Export params/deploy.yaml for an existing training run."""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export deploy.yaml for a training log directory.")
parser.add_argument("--task", type=str, required=True, help="Task name used for training.")
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Absolute or relative path to a run log directory (e.g. logs/rsl_rl/T800_flat/2026-06-01_12-24-09).",
)
parser.add_argument("--load_run", type=str, default=None, help="Run folder name under logs/rsl_rl/<experiment>.")
parser.add_argument("--motion_file", type=str, default=None, help="Optional motion .npz override.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.export_deploy_cfg import export_deploy_cfg


def _resolve_log_dir(experiment_name: str) -> str:
    if args_cli.log_dir is not None:
        log_dir = pathlib.Path(args_cli.log_dir).expanduser().resolve()
        if not log_dir.is_dir():
            raise FileNotFoundError(f"log_dir not found: {log_dir}")
        return str(log_dir)

    if args_cli.load_run is None:
        parser.error("Provide --log_dir or --load_run.")

    log_dir = pathlib.Path("logs") / "rsl_rl" / experiment_name / args_cli.load_run
    log_dir = log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")
    return str(log_dir)


def _load_motion_file_from_log(log_dir: str) -> str | None:
    env_yaml = os.path.join(log_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        return None
    with open(env_yaml, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("motion_file:"):
                motion_file = line.split(":", 1)[1].strip()
                motion_path = pathlib.Path(motion_file).expanduser()
                return str(motion_path.resolve()) if motion_path.is_file() else motion_file
    return None


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    log_dir = _resolve_log_dir(agent_cfg.experiment_name)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    motion_file = args_cli.motion_file
    if motion_file is None:
        motion_file = _load_motion_file_from_log(log_dir)
    if motion_file is None:
        motion_file = getattr(env_cfg.commands.motion, "motion_file", None)
    if motion_file is None or not pathlib.Path(motion_file).is_file():
        parser.error("Motion file missing. Pass --motion_file or ensure params/env.yaml records motion_file.")
    env_cfg.commands.motion.motion_file = str(pathlib.Path(motion_file).expanduser().resolve())

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    deploy_path = export_deploy_cfg(env.unwrapped, log_dir)
    print(f"[INFO] Exported deploy config to: {deploy_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
