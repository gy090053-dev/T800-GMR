#!/usr/bin/env python3
"""Convert T800 npy motion to npz format aligned with Isaac Sim body indexing.

Compared to ``npy_to_npz.py``:
- ``joint_pos`` / ``joint_vel`` stay 25-D and follow ``T800_MOTION_JOINT_NAMES``
  (compat URDF actuated joints, excluding fixed joints).
- ``body_*`` arrays follow the PhysX articulation body order (BFS over the URDF
  joint tree, skipping links merged by fixed joints). This matches
  ``robot.find_bodies(...)`` indices used in training, not the raw URDF
  ``<link>`` declaration order.

Usage::

    python scripts/repaired_npy_to_npz.py \\
        -i /path/to/motion.npy \\
        -o /path/to/motion_repaired.npz
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ast
import numpy as np

# ---------------------------------------------------------------------------
# T800 conventions (read from robots/t800.py without importing Isaac Lab)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_T800_PY = _REPO_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "robots" / "t800.py"
_ASSETS_DIR = _REPO_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "assets"
_COMPAT_URDF = _ASSETS_DIR / "t800" / "urdf" / "serial_t800_compat.urdf"
_RAW_URDF = _ASSETS_DIR / "t800" / "urdf" / "serial_t800.urdf"
T800_URDF_PATH = str(_COMPAT_URDF if _COMPAT_URDF.is_file() else _RAW_URDF)


def _load_motion_joint_names() -> list[str]:
    tree = ast.parse(_T800_PY.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "T800_MOTION_JOINT_NAMES":
                value = ast.literal_eval(node.value)
                if not isinstance(value, list):
                    raise RuntimeError("T800_MOTION_JOINT_NAMES must be a list in t800.py")
                return value
    raise RuntimeError(f"T800_MOTION_JOINT_NAMES not found in {_T800_PY}")


T800_MOTION_JOINT_NAMES = _load_motion_joint_names()

# Column order inside raw EngineAI npy files (legacy naming on right arm / head).
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

LEGACY_JOINT_TO_SIM = {
    legacy_name: sim_name
    for legacy_name, sim_name in zip(LEGACY_NPY_JOINT_NAMES, T800_MOTION_JOINT_NAMES, strict=True)
}
SIM_TO_LEGACY_JOINT = {sim_name: legacy_name for legacy_name, sim_name in LEGACY_JOINT_TO_SIM.items()}


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis_xyz: np.ndarray


def parse_vec3(text: Optional[str], default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def quat_normalize(quat_wxyz: np.ndarray) -> np.ndarray:
    return normalize(quat_wxyz)


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    result = quat_wxyz.copy()
    result[..., 1:] *= -1.0
    return result


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_from_axis_angle(axis_xyz: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis_xyz, dtype=np.float64))
    half = 0.5 * angle
    return np.asarray([math.cos(half), *(math.sin(half) * axis)], dtype=np.float64)


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * s,
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = np.asarray(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = np.asarray(
            [
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s,
                (rot[1, 2] + rot[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = np.asarray(
            [
                (rot[1, 0] - rot[0, 1]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
                (rot[1, 2] + rot[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat_normalize(quat)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, blend: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return quat_normalize((1.0 - blend) * q0 + blend * q1)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * blend
    s0 = math.sin(theta_0 - theta) / max(sin_theta_0, 1e-8)
    s1 = math.sin(theta) / max(sin_theta_0, 1e-8)
    return quat_normalize(s0 * q0 + s1 * q1)


def quat_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = quat_normalize(quat_wxyz)
    if quat[0] < 0.0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = xyz / sin_half
    angle = 2.0 * math.atan2(sin_half, w)
    return axis * angle


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def make_transform(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = pos
    return tf


def apply_joint(parent_tf: np.ndarray, spec: JointSpec, joint_angle: float) -> np.ndarray:
    origin_tf = make_transform(rpy_to_matrix(spec.origin_rpy), spec.origin_xyz)
    if spec.joint_type in {"revolute", "continuous"}:
        joint_rot = make_transform(quat_to_matrix(quat_from_axis_angle(spec.axis_xyz, joint_angle)), np.zeros(3))
    else:
        joint_rot = np.eye(4, dtype=np.float64)
    return parent_tf @ origin_tf @ joint_rot


def save_npz_non_zip64(output_path: Path, **arrays: np.ndarray) -> None:
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=False, compresslevel=6
    ) as zf:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            zf.writestr(f"{name}.npy", buffer.getvalue())


def resolve_urdf(explicit_urdf: Optional[str]) -> Path:
    if explicit_urdf:
        urdf_path = Path(explicit_urdf).expanduser().resolve()
        if urdf_path.exists():
            return urdf_path
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    urdf_path = Path(T800_URDF_PATH)
    if urdf_path.exists():
        return urdf_path
    raise FileNotFoundError(f"URDF not found: {urdf_path}")


def parse_urdf_link_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    link_names = [link.attrib["name"] for link in root.findall("link")]
    if not link_names:
        raise ValueError(f"No links found in URDF: {urdf_path}")
    return link_names


def parse_urdf_children(urdf_path: Path) -> dict[str, list[str]]:
    """Map each parent link to child links in URDF joint declaration order."""
    root = ET.parse(urdf_path).getroot()
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        children.setdefault(parent_elem.attrib["link"], []).append(child_elem.attrib["link"])
    return children


def parse_merged_link_names(urdf_path: Path) -> set[str]:
    """Return child links merged into their parent by fixed joints in PhysX."""
    root = ET.parse(urdf_path).getroot()
    merged: set[str] = set()
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "fixed") != "fixed":
            continue
        child_elem = joint.find("child")
        if child_elem is not None:
            merged.add(child_elem.attrib["link"])
    return merged


def parse_physx_body_names(urdf_path: Path, root_link: str = "LINK_BASE") -> list[str]:
    """Return Isaac Sim / PhysX rigid-body names in BFS articulation order.

    Fixed-joint child links (e.g. feet and wrist ends) are omitted because PhysX
    merges them into their parent bodies. The resulting list can be indexed
    directly with ``robot.find_bodies(name)`` during training.
    """
    merged_links = parse_merged_link_names(urdf_path)
    children = parse_urdf_children(urdf_path)

    body_names: list[str] = []
    queue: deque[str] = deque([root_link])
    seen: set[str] = set()
    while queue:
        link_name = queue.popleft()
        if link_name in seen or link_name in merged_links:
            continue
        seen.add(link_name)
        body_names.append(link_name)
        for child_name in children.get(link_name, []):
            if child_name not in merged_links and child_name not in seen:
                queue.append(child_name)

    if root_link not in body_names:
        raise ValueError(f"Root link '{root_link}' not found in URDF articulation tree: {urdf_path}")
    return body_names


def reorder_link_arrays_to_physx(
    link_positions: np.ndarray,
    link_quaternions: np.ndarray,
    urdf_link_names: list[str],
    physx_body_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Gather FK link arrays into PhysX body column order."""
    link_index = {name: idx for idx, name in enumerate(urdf_link_names)}
    missing = [name for name in physx_body_names if name not in link_index]
    if missing:
        raise KeyError(f"PhysX body names missing from URDF FK output: {missing}")

    body_pos = np.empty((link_positions.shape[0], len(physx_body_names), 3), dtype=np.float32)
    body_quat = np.empty((link_quaternions.shape[0], len(physx_body_names), 4), dtype=np.float32)
    for out_idx, body_name in enumerate(physx_body_names):
        src_idx = link_index[body_name]
        body_pos[:, out_idx] = link_positions[:, src_idx]
        body_quat[:, out_idx] = link_quaternions[:, src_idx]
    return body_pos, body_quat


def parse_urdf_joints(urdf_path: Path) -> dict[str, JointSpec]:
    root = ET.parse(urdf_path).getroot()
    incoming: dict[str, JointSpec] = {}
    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        origin_elem = joint.find("origin")
        axis_elem = joint.find("axis")
        spec = JointSpec(
            name=joint.attrib["name"],
            joint_type=joint.attrib.get("type", "fixed"),
            parent_link=parent_elem.attrib["link"],
            child_link=child_elem.attrib["link"],
            origin_xyz=parse_vec3(None if origin_elem is None else origin_elem.attrib.get("xyz"), (0.0, 0.0, 0.0)),
            origin_rpy=parse_vec3(None if origin_elem is None else origin_elem.attrib.get("rpy"), (0.0, 0.0, 0.0)),
            axis_xyz=parse_vec3(None if axis_elem is None else axis_elem.attrib.get("xyz"), (1.0, 0.0, 0.0)),
        )
        incoming[spec.child_link] = spec
    return incoming


def resample_motion(
    raw_motion: np.ndarray, input_fps: float, output_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw_motion.ndim != 2 or raw_motion.shape[1] < 32:
        raise ValueError(f"Expected motion shape [T, >=32], got {raw_motion.shape}")

    base_pos = raw_motion[:, 0:3].astype(np.float64)
    base_quat = raw_motion[:, 3:7].astype(np.float64)
    legacy_joint_pos = raw_motion[:, 7:32].astype(np.float64)

    for i in range(1, base_quat.shape[0]):
        if np.dot(base_quat[i - 1], base_quat[i]) < 0.0:
            base_quat[i] = -base_quat[i]

    if raw_motion.shape[0] <= 1:
        return (
            base_pos.astype(np.float32),
            quat_normalize(base_quat).astype(np.float32),
            legacy_joint_pos.astype(np.float32),
        )

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (raw_motion.shape[0] - 1) * input_dt
    sample_times = np.arange(0.0, duration, output_dt, dtype=np.float64)
    if sample_times.size == 0:
        sample_times = np.asarray([0.0], dtype=np.float64)

    phase = sample_times / max(duration, 1e-8)
    scaled = phase * (raw_motion.shape[0] - 1)
    idx0 = np.floor(scaled).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, raw_motion.shape[0] - 1)
    blend = scaled - idx0

    out_base_pos = (1.0 - blend[:, None]) * base_pos[idx0] + blend[:, None] * base_pos[idx1]
    out_joint_pos = (1.0 - blend[:, None]) * legacy_joint_pos[idx0] + blend[:, None] * legacy_joint_pos[idx1]
    out_base_quat = np.stack(
        [quat_slerp(base_quat[a], base_quat[b], float(t)) for a, b, t in zip(idx0, idx1, blend, strict=False)],
        axis=0,
    )
    return out_base_pos.astype(np.float32), out_base_quat.astype(np.float32), out_joint_pos.astype(np.float32)


def remap_legacy_joints_to_sim(legacy_joint_pos: np.ndarray) -> np.ndarray:
    """Map legacy npy joint columns onto ``T800_MOTION_JOINT_NAMES`` order."""
    if legacy_joint_pos.shape[1] != len(T800_MOTION_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(T800_MOTION_JOINT_NAMES)} legacy joint columns, got {legacy_joint_pos.shape[1]}"
        )
    legacy_index = {name: idx for idx, name in enumerate(LEGACY_NPY_JOINT_NAMES)}
    sim_joint_pos = np.empty((legacy_joint_pos.shape[0], len(T800_MOTION_JOINT_NAMES)), dtype=np.float32)
    for out_idx, sim_name in enumerate(T800_MOTION_JOINT_NAMES):
        legacy_name = SIM_TO_LEGACY_JOINT[sim_name]
        sim_joint_pos[:, out_idx] = legacy_joint_pos[:, legacy_index[legacy_name]]
    return sim_joint_pos


def forward_kinematics_urdf_order(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    incoming_joints: dict[str, JointSpec],
    link_names: list[str],
    motion_joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    frames = joint_pos.shape[0]
    num_links = len(link_names)
    joint_index = {name: idx for idx, name in enumerate(motion_joint_names)}

    link_positions = np.zeros((frames, num_links, 3), dtype=np.float32)
    link_quaternions = np.zeros((frames, num_links, 4), dtype=np.float32)

    for frame_idx in range(frames):
        cache: dict[str, np.ndarray] = {
            "LINK_BASE": make_transform(quat_to_matrix(base_quat[frame_idx]), base_pos[frame_idx])
        }

        def world_transform(link_name: str) -> np.ndarray:
            if link_name in cache:
                return cache[link_name]
            if link_name not in incoming_joints:
                raise KeyError(f"Link '{link_name}' not found in URDF joint tree")
            spec = incoming_joints[link_name]
            parent_tf = world_transform(spec.parent_link)
            if spec.joint_type in {"revolute", "continuous"} and spec.name in joint_index:
                angle = float(joint_pos[frame_idx, joint_index[spec.name]])
            else:
                angle = 0.0
            cache[link_name] = apply_joint(parent_tf, spec, angle)
            return cache[link_name]

        for link_idx, link_name in enumerate(link_names):
            tf = world_transform(link_name)
            link_positions[frame_idx, link_idx] = tf[:3, 3].astype(np.float32)
            link_quaternions[frame_idx, link_idx] = matrix_to_quat_wxyz(tf[:3, :3]).astype(np.float32)

    for link_idx in range(link_quaternions.shape[1]):
        for frame_idx in range(1, link_quaternions.shape[0]):
            if np.dot(link_quaternions[frame_idx - 1, link_idx], link_quaternions[frame_idx, link_idx]) < 0.0:
                link_quaternions[frame_idx, link_idx] *= -1.0

    return link_positions, link_quaternions


def compute_linear_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    if positions.shape[0] <= 1:
        return np.zeros_like(positions, dtype=np.float32)
    return np.gradient(positions, dt, axis=0).astype(np.float32)


def compute_angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    frames, num_bodies, _ = quaternions.shape
    angular_velocity = np.zeros((frames, num_bodies, 3), dtype=np.float32)
    if frames <= 1:
        return angular_velocity
    if frames == 2:
        rel = quat_mul(quaternions[1], quat_conjugate(quaternions[0]))
        omega = np.stack([quat_to_axis_angle(rel[body_idx]) / dt for body_idx in range(num_bodies)], axis=0)
        angular_velocity[:] = omega.astype(np.float32)
        return angular_velocity

    for body_idx in range(num_bodies):
        for frame_idx in range(1, frames - 1):
            rel = quat_mul(quaternions[frame_idx + 1, body_idx], quat_conjugate(quaternions[frame_idx - 1, body_idx]))
            angular_velocity[frame_idx, body_idx] = (quat_to_axis_angle(rel) / (2.0 * dt)).astype(np.float32)
        angular_velocity[0, body_idx] = angular_velocity[1, body_idx]
        angular_velocity[-1, body_idx] = angular_velocity[-2, body_idx]
    return angular_velocity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert T800 npy motion to URDF-aligned npz for whole_body_tracking"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .npy file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .npz file")
    parser.add_argument("--fps", type=float, default=50.0, help="Output fps (default 50)")
    parser.add_argument("--input_fps", type=float, default=30.0, help="Input npy fps (default 30)")
    parser.add_argument("--urdf", type=str, default=None, help="Path to T800 URDF; defaults to compat URDF from t800.py")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    urdf_path = resolve_urdf(args.urdf)
    link_names = parse_urdf_link_names(urdf_path)
    incoming_joints = parse_urdf_joints(urdf_path)

    raw_motion = np.load(input_path, allow_pickle=False)
    if raw_motion.ndim == 1:
        raw_motion = np.asarray(raw_motion.tolist(), dtype=np.float32)

    base_pos, base_quat, legacy_joint_pos = resample_motion(raw_motion, args.input_fps, args.fps)
    joint_pos = remap_legacy_joints_to_sim(legacy_joint_pos)
    dt = 1.0 / args.fps
    joint_vel = compute_linear_velocity(joint_pos, dt)

    link_positions, link_quaternions = forward_kinematics_urdf_order(
        base_pos,
        base_quat,
        joint_pos,
        incoming_joints,
        link_names,
        T800_MOTION_JOINT_NAMES,
    )
    physx_body_names = parse_physx_body_names(urdf_path)
    body_pos_w, body_quat_w = reorder_link_arrays_to_physx(
        link_positions,
        link_quaternions,
        link_names,
        physx_body_names,
    )
    body_lin_vel = compute_linear_velocity(body_pos_w, dt)
    body_ang_vel = compute_angular_velocity(body_quat_w, dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_non_zip64(
        output_path,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        fps=np.asarray([args.fps], dtype=np.float32),
        body_names=np.asarray(physx_body_names),
    )

    merged_links = sorted(parse_merged_link_names(urdf_path))
    print(f"[repaired_npy_to_npz] URDF: {urdf_path}")
    print(f"[repaired_npy_to_npz] Input: {input_path}")
    print(f"[repaired_npy_to_npz] Output: {output_path}")
    print(
        f"[repaired_npy_to_npz] Frames: {joint_pos.shape[0]}, "
        f"Joints: {joint_pos.shape[1]}, Bodies: {len(physx_body_names)} "
        f"(URDF links: {len(link_names)}, merged: {len(merged_links)})"
    )
    print(f"[repaired_npy_to_npz] Joint order: T800_MOTION_JOINT_NAMES")
    print(
        f"[repaired_npy_to_npz] Body order: PhysX BFS "
        f"({physx_body_names[0]} ... {physx_body_names[-1]})"
    )
    if merged_links:
        print(f"[repaired_npy_to_npz] Merged fixed links: {', '.join(merged_links)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
