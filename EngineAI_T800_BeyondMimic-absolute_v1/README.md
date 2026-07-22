# Whole Body Tracking EngineAI Release Notes

This release folder contains a public, self-contained description of the
`T800` motion format used by this repository.

The original repository script `scripts/npy_to_npz.py` depended on an external
private checkout named `engineaimuaythailab` to import
`T800MotionLoader`. The release version in this folder removes that code
dependency and reconstructs the required motion tensors from:

- the raw `npy` motion array,
- the public `T800` URDF already shipped in this repository, and
- the body/joint naming used by `whole_body_tracking`.

## Files

- `scripts/npy_to_npz.py`: public converter from EngineAI-style `npy` to the
  `npz` format consumed by `whole_body_tracking`.

## Input NPY Format

Each frame is expected to contain at least 32 columns:

- `0:3`: base position `(x, y, z)`
- `3:7`: base quaternion `(w, x, y, z)`
- `7:32`: 25 joint positions in the following order:
  - `J00_HIP_PITCH_L`
  - `J01_HIP_ROLL_L`
  - `J02_HIP_YAW_L`
  - `J03_KNEE_PITCH_L`
  - `J04_ANKLE_PITCH_L`
  - `J05_ANKLE_ROLL_L`
  - `J06_HIP_PITCH_R`
  - `J07_HIP_ROLL_R`
  - `J08_HIP_YAW_R`
  - `J09_KNEE_PITCH_R`
  - `J10_ANKLE_PITCH_R`
  - `J11_ANKLE_ROLL_R`
  - `J12_TORSO_YAW`
  - `J13_SHOULDER_PITCH_L`
  - `J14_SHOULDER_ROLL_L`
  - `J15_SHOULDER_YAW_L`
  - `J16_ELBOW_PITCH_L`
  - `J17_ELBOW_YAW_L`
  - `J20_SHOULDER_PITCH_R`
  - `J21_SHOULDER_ROLL_R`
  - `J22_SHOULDER_YAW_R`
  - `J23_ELBOW_PITCH_R`
  - `J24_ELBOW_YAW_R`
  - `J27_HEAD_PITCH`
  - `J28_HEAD_YAW`
- `32:`: optional extra channels such as contacts; ignored by the converter

## Output NPZ Format

The generated `npz` contains the tensors expected by
`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`:

- `joint_pos`: `[T, 25]`
- `joint_vel`: `[T, 25]`
- `body_pos_w`: `[T, B, 3]`
- `body_quat_w`: `[T, B, 4]`, quaternion order `(w, x, y, z)`
- `body_lin_vel_w`: `[T, B, 3]`
- `body_ang_vel_w`: `[T, B, 3]`
- `fps`: `[1]`

## Usage

Run from the repository root:

```bash
python whole_body_tracking_engineai_release/whole_body_tracking_engineai/scripts/npy_to_npz.py \
  -i /path/to/motion.npy \
  -o /path/to/motion.npz \
  --use_dfs
```

The converter auto-locates the public `T800` URDF from either the repository
layout or this release package. If your package layout is different, pass the
URDF path explicitly:

```bash
python scripts/npy_to_npz.py \
  -i motion.npy \
  -o motion.npz \
  --urdf /path/to/serial_t800.urdf \
  --use_dfs
```

## Body Order Notes

- `--use_dfs` matches `T800_MOTION_BODY_NAMES` used by this repository.
- The default order is BFS to stay compatible with the older internal script.
- `--body_order_file` can load a Python file that defines `T800_BODY_ORDER`.

## Reproducibility Notes

This public converter reproduces the repository-facing `npz` contract without
calling external private code. It does:

- temporal resampling,
- joint velocity estimation,
- URDF-based forward kinematics for the public `T800` chain,
- linear velocity estimation from positions, and
- angular velocity estimation from quaternion differences.

The following compatibility behavior matches the older converter:

- toe and heel bodies reuse the parent ankle transform,
- wrist bodies reuse the parent elbow-yaw transform.

If the private loader applied extra heuristics outside this repository
(for example custom base corrections), results may differ slightly. The output
is still directly usable by the public `whole_body_tracking` loader in this
repository.
