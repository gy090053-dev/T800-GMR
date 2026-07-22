import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _get_policy_and_normalizer(runner: OnPolicyRunner):
    """Resolve policy and observation normalizer across rsl-rl versions."""
    if hasattr(runner.alg, "get_policy"):
        policy = runner.alg.get_policy()
        normalizer = getattr(policy, "obs_normalizer", None)
    else:
        policy = runner.alg.policy
        normalizer = getattr(runner, "obs_normalizer", None)
    return policy, normalizer


def _get_logger_type(runner: OnPolicyRunner) -> str:
    """Resolve logger type across rsl-rl versions."""
    if hasattr(runner, "logger") and hasattr(runner.logger, "logger_type"):
        return runner.logger.logger_type
    return getattr(runner, "logger_type", "")


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _get_logger_type(self) in ["wandb"]:
            policy, normalizer = _get_policy_and_normalizer(self)
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _get_logger_type(self) in ["wandb"]:
            policy, normalizer = _get_policy_and_normalizer(self)
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
