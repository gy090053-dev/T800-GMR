"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --robot t800 --input_file ./data/source_motion.csv --input_fps 30 \
    --output_file ./data/source_motion_50hz.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument(
    "--robot", type=str, default="pm01", choices=["pm01", "t800"], help="The robot configuration to use."
)
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file")
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help=(
        "The path to the output motion npz file. If not provided, the output file will have the same name as the input"
        " file with .npz extension."
    ),
)
parser.add_argument("--input_fps", type=int, default=50, help="The fps of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--wandb", action="store_true", help="Whether to log to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="csv_to_npz", help="The name of the wandb project.")
parser.add_argument("--wandb_registry", type=str, default="motions", help="The wandb artifact type to link motion files under.")
parser.add_argument(
    "--wandb_collection",
    type=str,
    required=False,
    help="The wandb collection name. If not provided, the collection name will be the input file stem.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

assert (
    Path(args_cli.input_file).resolve() != Path(args_cli.output_file).resolve()
), "Input and output file paths must be different."

# launch omniverse app
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

##
# Pre-defined configs
##
from whole_body_tracking.robots.pm01 import PM01_CYLINDER_CFG
from whole_body_tracking.robots.t800 import T800_CFG

SOURCE_JOINT_ORDERS = {
    "pm01": [
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
        "J12_WAIST_YAW",
        "J13_SHOULDER_PITCH_L",
        "J14_SHOULDER_ROLL_L",
        "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L",
        "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R",
        "J19_SHOULDER_ROLL_R",
        "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R",
        "J22_ELBOW_YAW_R",
        "J23_HEAD_YAW",
    ],
    "t800": [
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
    ],
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    if args_cli.robot == "pm01":
        robot: ArticulationCfg = PM01_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.robot == "t800":
        robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_npz_motion(self):
        with np.load(self.motion_file) as f:
            data = {k: f[k].copy() for k in f.files}

        start = None if self.frame_range is None else self.frame_range[0]
        end = None if self.frame_range is None else self.frame_range[1] + 1
        root_pos = data["root_pos"][start:end]
        root_rot = data["root_rot"][start:end]
        dof_pos = data["dof_pos"][start:end]
        return np.concatenate((root_pos, root_rot, dof_pos), axis=1)

    def _load_pkl_motion(self):
        import pickle

        with open(self.motion_file, "rb") as f:
            data = pickle.load(f)

            start = None if self.frame_range is None else self.frame_range[0]
            end = None if self.frame_range is None else self.frame_range[1] + 1
            root_pos = data["root_pos"][start:end]
            root_rot = data["root_rot"][start:end]
            dof_pos = data["dof_pos"][start:end]
        return np.concatenate((root_pos, root_rot, dof_pos), axis=1)

    def _load_csv_motion(self):
        if self.frame_range is None:
            return np.loadtxt(self.motion_file, delimiter=",", skiprows=1)  # skip head
        else:
            return np.loadtxt(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )

    def _load_motion(self):
        """Loads the motion from the csv file."""
        # load motion (support .pkl and .csv, case-insensitive)
        if self.motion_file.lower().endswith(".pkl"):
            motion = self._load_pkl_motion()
        elif self.motion_file.lower().endswith(".csv"):
            motion = self._load_csv_motion()
        else:
            motion = self._load_npz_motion()

        motion = torch.from_numpy(motion).to(dtype=torch.float32).to(device=self.device)

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7 : 7 + 24]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
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
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
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

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            if args_cli.wandb:
                np.savez("/tmp/motion.npz", **log)

                import wandb

                PROJECT = args_cli.wandb_project if args_cli.wandb_project else "csv_to_npz"
                REGISTRY = args_cli.wandb_registry if args_cli.wandb_registry else "motions"
                COLLECTION = args_cli.wandb_collection if args_cli.wandb_collection else Path(args_cli.input_file).stem
                run = wandb.init(project=PROJECT, name=COLLECTION)
                print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
                logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
                run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
                print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
            else:
                input_path = Path(args_cli.input_file)
                output_path = Path(args_cli.output_file) if args_cli.output_file else input_path.with_suffix(".npz")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Logging motion to {output_path}")
                # breakpoint()
                np.savez(output_path, **log)  # type: ignore[arg-type]

                # csv_path = output_path.with_suffix(".csv")
                # import pandas as pd
                # df = pd.DataFrame(log)
                # df.to_csv(csv_path, index=True)
                print(f"[INFO] Motion saved to {output_path}.")

            return


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=SOURCE_JOINT_ORDERS[args_cli.robot],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # TODO: implement proper cleanup
    # This is an unresolved bug in isaacsim 5.0（https://github.com/isaac-sim/IsaacLab/issues/3475);
    # Here is a temporary fix.
    raise KeyboardInterrupt
    simulation_app.close()
