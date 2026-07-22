"""Replay motion from CSV (or pkl/npz) and export npz aligned with T800 training.

Compared to ``csv_to_npz.py`` (T800 path):
- Drives the robot with ``T800_MOTION_JOINT_NAMES`` (25 actuated joints, compat URDF).
- Remaps legacy 24-DOF CSV columns (``SOURCE_JOINT_ORDERS`` style: no torso/head,
  right-arm ``J20``–``J24`` naming) onto sim joint names.
- Logs ``joint_pos`` / ``joint_vel`` as ``[T, 25]`` in motion order, not the full
  articulation vector.
- Logs ``body_*`` in PhysX articulation order (same layout as Isaac Sim replay).
- Writes ``body_names`` into the npz for traceability.

.. code-block:: bash

    python scripts/repaired_csv_to_npz.py \\
        --input_file ./data/source_motion.csv --input_fps 30 \\
        --output_file ./data/source_motion_repaired.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WBT_SOURCE = _REPO_ROOT / "source" / "whole_body_tracking"
if _WBT_SOURCE.is_dir() and str(_WBT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_WBT_SOURCE))

parser = argparse.ArgumentParser(
    description="Replay motion from csv file and output training-aligned npz (T800 repaired)."
)
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file")
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="The path to the output motion npz file.",
)
parser.add_argument("--input_fps", type=int, default=50, help="The fps of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. "
        "If not provided, all frames will be loaded."
    ),
)
parser.add_argument(
    "--input_joint_order",
    type=str,
    default="auto",
    choices=["auto", "legacy_csv", "legacy_npy", "sim"],
    help=(
        "Joint column semantics after root pose (7 columns). "
        "'auto': 24 cols -> legacy_csv, 25 cols -> sim (T800_MOTION_JOINT_NAMES), "
        "31+ cols -> legacy_npy. Override when your CSV header order is known."
    ),
)
parser.add_argument("--wandb", action="store_true", help="Whether to log to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="repaired_csv_to_npz", help="The wandb project name.")
parser.add_argument("--wandb_registry", type=str, default="motions", help="The wandb artifact registry type.")
parser.add_argument(
    "--wandb_collection",
    type=str,
    required=False,
    help="The wandb collection name. Defaults to the input file stem.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

assert (
    Path(args_cli.input_file).resolve() != Path(args_cli.output_file).resolve()
), "Input and output file paths must be different."

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.t800 import T800_CFG, T800_DEFAULT_JOINT_POS, T800_MOTION_JOINT_NAMES

# Legacy 24-DOF CSV column order from csv_to_npz.py (no torso yaw, no head, J20–J24 right arm).
LEGACY_CSV_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
]

# Legacy 25-DOF EngineAI npy column order (torso + head, J20–J24 right arm naming).
LEGACY_NPY_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
]

LEGACY_CSV_TO_SIM: dict[str, str] = {
    name: name for name in LEGACY_CSV_JOINT_NAMES if name in T800_MOTION_JOINT_NAMES
}
LEGACY_CSV_TO_SIM.update(
    {
        "J20_SHOULDER_PITCH_R": "J18_SHOULDER_PITCH_R",
        "J21_SHOULDER_ROLL_R": "J19_SHOULDER_ROLL_R",
        "J22_SHOULDER_YAW_R": "J20_SHOULDER_YAW_R",
        "J23_ELBOW_PITCH_R": "J21_ELBOW_PITCH_R",
        "J24_ELBOW_YAW_R": "J22_ELBOW_YAW_R",
    }
)

LEGACY_NPY_TO_SIM = {
    legacy_name: sim_name
    for legacy_name, sim_name in zip(LEGACY_NPY_JOINT_NAMES, T800_MOTION_JOINT_NAMES, strict=True)
}

def _default_t800_sim_joint_pos(device: torch.device, frames: int) -> torch.Tensor:
    defaults = torch.tensor(
        [T800_DEFAULT_JOINT_POS[name] for name in T800_MOTION_JOINT_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return defaults.unsqueeze(0).expand(frames, -1).clone()


def _infer_joint_order(num_dof: int, joint_order: str) -> str:
    if joint_order != "auto":
        return joint_order
    if num_dof == len(LEGACY_CSV_JOINT_NAMES):
        return "legacy_csv"
    if num_dof == len(T800_MOTION_JOINT_NAMES):
        return "sim"
    if num_dof == len(LEGACY_NPY_JOINT_NAMES):
        return "legacy_npy"
    raise ValueError(
        f"Cannot infer joint order for {num_dof} DOF columns. "
        f"Expected {len(LEGACY_CSV_JOINT_NAMES)} (legacy_csv), "
        f"{len(T800_MOTION_JOINT_NAMES)} (sim), or {len(LEGACY_NPY_JOINT_NAMES)} (legacy_npy). "
        "Pass --input_joint_order explicitly."
    )


def _remap_dof_to_sim(dof_pos: torch.Tensor, order: str) -> torch.Tensor:
    """Map input DOF columns to ``T800_MOTION_JOINT_NAMES`` order."""
    frames = dof_pos.shape[0]
    sim_pos = _default_t800_sim_joint_pos(dof_pos.device, frames)

    if order == "sim":
        if dof_pos.shape[1] != len(T800_MOTION_JOINT_NAMES):
            raise ValueError(f"Expected {len(T800_MOTION_JOINT_NAMES)} sim joint columns, got {dof_pos.shape[1]}")
        return dof_pos

    if order == "legacy_csv":
        legacy_index = {name: idx for idx, name in enumerate(LEGACY_CSV_JOINT_NAMES)}
        sim_index = {name: idx for idx, name in enumerate(T800_MOTION_JOINT_NAMES)}
        for legacy_name, sim_name in LEGACY_CSV_TO_SIM.items():
            sim_pos[:, sim_index[sim_name]] = dof_pos[:, legacy_index[legacy_name]]
        return sim_pos

    if order == "legacy_npy":
        legacy_index = {name: idx for idx, name in enumerate(LEGACY_NPY_JOINT_NAMES)}
        sim_index = {name: idx for idx, name in enumerate(T800_MOTION_JOINT_NAMES)}
        for legacy_name, sim_name in LEGACY_NPY_TO_SIM.items():
            sim_pos[:, sim_index[sim_name]] = dof_pos[:, legacy_index[legacy_name]]
        return sim_pos

    raise ValueError(f"Unsupported joint order: {order}")


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

    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        input_joint_order: str,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.input_joint_order = input_joint_order
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_csv_motion(self) -> np.ndarray:
        data = np.loadtxt(self.motion_file, delimiter=",", skiprows=1)
        if self.frame_range is not None:
            start = self.frame_range[0] - 1
            end = self.frame_range[1]
            data = data[start:end]
        return data

    def _load_npz_motion(self) -> np.ndarray:
        with np.load(self.motion_file) as f:
            data = {k: f[k].copy() for k in f.files}

        start = None if self.frame_range is None else self.frame_range[0]
        end = None if self.frame_range is None else self.frame_range[1] + 1
        root_pos = data["root_pos"][start:end]
        root_rot = data["root_rot"][start:end]
        dof_pos = data["dof_pos"][start:end]
        return np.concatenate((root_pos, root_rot, dof_pos), axis=1)

    def _load_pkl_motion(self) -> np.ndarray:
        import pickle

        with open(self.motion_file, "rb") as f:
            data = pickle.load(f)

        start = None if self.frame_range is None else self.frame_range[0]
        end = None if self.frame_range is None else self.frame_range[1] + 1
        root_pos = data["root_pos"][start:end]
        root_rot = data["root_rot"][start:end]
        dof_pos = data["dof_pos"][start:end]
        return np.concatenate((root_pos, root_rot, dof_pos), axis=1)

    def _load_motion(self) -> None:
        if self.motion_file.lower().endswith(".pkl"):
            motion = self._load_pkl_motion()
        elif self.motion_file.lower().endswith(".csv"):
            motion = self._load_csv_motion()
        else:
            motion = self._load_npz_motion()

        motion = torch.from_numpy(motion).to(dtype=torch.float32).to(device=self.device)

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        input_dof = motion[:, 7:]

        order = _infer_joint_order(input_dof.shape[1], self.input_joint_order)
        self.motion_dof_poss_input = _remap_dof_to_sim(input_dof, order)
        print(f"[repaired_csv_to_npz] Input DOF: {input_dof.shape[1]}, joint order: {order}")

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"[repaired_csv_to_npz] Motion loaded ({self.motion_file}), "
            f"duration: {self.duration:.3f} s, frames: {self.input_frames}"
        )

    def _interpolate_motion(self) -> None:
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"[repaired_csv_to_npz] Interpolated: {self.input_frames} @ {self.input_fps} Hz "
            f"-> {self.output_frames} @ {self.output_fps} Hz"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        bool,
    ]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    joint_names = T800_MOTION_JOINT_NAMES
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        input_joint_order=args_cli.input_joint_order,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    if len(robot_joint_indexes) != len(joint_names):
        raise ValueError(
            f"Robot joint count mismatch: found {len(robot_joint_indexes)} for "
            f"{len(joint_names)} motion joints ({joint_names[:3]} ...)."
        )

    log: dict[str, list | list[float]] = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, robot_joint_indexes].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[key] = np.stack(log[key], axis=0)

            log["body_names"] = np.asarray(list(robot.body_names))

            if args_cli.wandb:
                np.savez("/tmp/motion.npz", **log)

                import wandb

                project = args_cli.wandb_project or "repaired_csv_to_npz"
                registry = args_cli.wandb_registry or "motions"
                collection = args_cli.wandb_collection or Path(args_cli.input_file).stem
                run = wandb.init(project=project, name=collection)
                print(f"[repaired_csv_to_npz] Logging motion to wandb: {collection}")
                logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=collection, type=registry)
                run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{registry}/{collection}")
                print(f"[repaired_csv_to_npz] Motion saved to wandb registry: {registry}/{collection}")
            else:
                output_path = Path(args_cli.output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[repaired_csv_to_npz] Saving motion to {output_path}")
                np.savez(output_path, **log)
                print(
                    f"[repaired_csv_to_npz] Saved: frames={log['joint_pos'].shape[0]}, "
                    f"joints={log['joint_pos'].shape[1]}, bodies={log['body_pos_w'].shape[1]}"
                )
                print("[repaired_csv_to_npz] Joint order: T800_MOTION_JOINT_NAMES")
                print(
                    f"[repaired_csv_to_npz] Body order: PhysX "
                    f"({robot.body_names[0]} ... {robot.body_names[-1]})"
                )
            return


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[repaired_csv_to_npz] Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    raise KeyboardInterrupt
    simulation_app.close()
