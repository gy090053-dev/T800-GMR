#!/usr/bin/env python3
"""Convert T800 npy motion to public npz format.

This version does not depend on the private `engineaimuaythailab` repository.
It reconstructs body states directly from the public T800 URDF.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


T800_DFS_JOINT_NAMES = [
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

T800_BODY_ORDER_BFS = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_PITCH_R",
    "LINK_TORSO_YAW",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_HEAD_PITCH",
    "LINK_HIP_YAW_L",
    "LINK_HIP_YAW_R",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_HEAD_YAW",
    "LINK_KNEE_PITCH_L",
    "LINK_KNEE_PITCH_R",
    "LINK_SHOULDER_YAW_L",
    "LINK_SHOULDER_YAW_R",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_PITCH_R",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_PITCH_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
    "LINK_ANKLE_ROLL_L_TOE",
    "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_ANKLE_ROLL_R_TOE",
    "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_WRIST_PITCH_L",
    "LINK_WRIST_PITCH_R",
    "LINK_WRIST_ROLL_L",
    "LINK_WRIST_ROLL_R",
]

T800_BODY_ORDER_DFS = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_TOE",
    "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_TOE",
    "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_L",
    "LINK_WRIST_ROLL_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_PITCH_R",
    "LINK_WRIST_ROLL_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]

AVAILABLE_LINK_NAMES = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]

CHILD_TO_PARENT = {
    "LINK_ANKLE_ROLL_L_TOE": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_HEEL": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R_TOE": "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_HEEL": "LINK_ANKLE_ROLL_R",
    "LINK_WRIST_PITCH_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_ROLL_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_R": "LINK_ELBOW_YAW_R",
    "LINK_WRIST_ROLL_R": "LINK_ELBOW_YAW_R",
}


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

    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parent.parent
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "assets"
        / "t800"
        / "urdf"
        / "serial_t800.urdf",
        script_path.parents[3]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "assets"
        / "t800"
        / "urdf"
        / "serial_t800.urdf",
        script_path.parent.parent / "assets" / "serial_t800.urdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not locate T800 URDF.\nTried:\n{tried}")


def load_body_order(body_order_file: Optional[str], use_dfs: bool) -> list[str]:
    if body_order_file is None:
        return T800_BODY_ORDER_DFS if use_dfs else T800_BODY_ORDER_BFS
    body_file = Path(body_order_file).expanduser().resolve()
    if not body_file.exists():
        raise FileNotFoundError(f"Body order file not found: {body_file}")
    namespace: dict[str, object] = {}
    exec(body_file.read_text(), namespace)
    body_order = namespace.get("T800_BODY_ORDER")
    if not isinstance(body_order, list) or not all(isinstance(name, str) for name in body_order):
        raise ValueError(f"{body_file} must define T800_BODY_ORDER as a list[str]")
    return body_order


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
    joint_pos = raw_motion[:, 7:32].astype(np.float64)

    for i in range(1, base_quat.shape[0]):
        if np.dot(base_quat[i - 1], base_quat[i]) < 0.0:
            base_quat[i] = -base_quat[i]

    if raw_motion.shape[0] <= 1:
        return base_pos.astype(np.float32), quat_normalize(base_quat).astype(np.float32), joint_pos.astype(np.float32)

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
    out_joint_pos = (1.0 - blend[:, None]) * joint_pos[idx0] + blend[:, None] * joint_pos[idx1]
    out_base_quat = np.stack(
        [quat_slerp(base_quat[a], base_quat[b], float(t)) for a, b, t in zip(idx0, idx1, blend, strict=False)], axis=0
    )
    return out_base_pos.astype(np.float32), out_base_quat.astype(np.float32), out_joint_pos.astype(np.float32)


def forward_kinematics(
    base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray, incoming_joints: dict[str, JointSpec]
) -> tuple[np.ndarray, np.ndarray]:
    frames = joint_pos.shape[0]
    link_positions = np.zeros((frames, len(AVAILABLE_LINK_NAMES), 3), dtype=np.float32)
    link_quaternions = np.zeros((frames, len(AVAILABLE_LINK_NAMES), 4), dtype=np.float32)
    joint_index = {name: idx for idx, name in enumerate(T800_DFS_JOINT_NAMES)}

    for frame_idx in range(frames):
        cache: dict[str, np.ndarray] = {
            "LINK_BASE": make_transform(quat_to_matrix(base_quat[frame_idx]), base_pos[frame_idx])
        }

        def world_transform(link_name: str) -> np.ndarray:
            if link_name in cache:
                return cache[link_name]
            if link_name not in incoming_joints:
                raise KeyError(f"Link '{link_name}' not found in T800 URDF joint tree")
            spec = incoming_joints[link_name]
            parent_tf = world_transform(spec.parent_link)
            angle = float(joint_pos[frame_idx, joint_index[spec.name]]) if spec.name in joint_index else 0.0
            cache[link_name] = apply_joint(parent_tf, spec, angle)
            return cache[link_name]

        for link_idx, link_name in enumerate(AVAILABLE_LINK_NAMES):
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


def build_body_arrays(
    body_order: list[str],
    link_positions: np.ndarray,
    link_quaternions: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    link_index = {name: idx for idx, name in enumerate(AVAILABLE_LINK_NAMES)}
    frames = link_positions.shape[0]
    num_bodies = len(body_order)

    body_pos = np.zeros((frames, num_bodies, 3), dtype=np.float32)
    body_quat = np.zeros((frames, num_bodies, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0

    for body_idx, body_name in enumerate(body_order):
        source_name = body_name
        if source_name not in link_index:
            source_name = CHILD_TO_PARENT.get(source_name, "LINK_BASE")
        source_idx = link_index.get(source_name, 0)
        body_pos[:, body_idx] = link_positions[:, source_idx]
        body_quat[:, body_idx] = link_quaternions[:, source_idx]

    body_lin_vel = compute_linear_velocity(body_pos, dt)
    body_ang_vel = compute_angular_velocity(body_quat, dt)
    return body_pos, body_quat, body_lin_vel, body_ang_vel


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert T800 npy motion to npz format for whole_body_tracking")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .npy file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .npz file")
    parser.add_argument("--fps", type=float, default=50.0, help="Output fps (default 50)")
    parser.add_argument("--input_fps", type=float, default=30.0, help="Input npy fps (default 30)")
    parser.add_argument("--use_dfs", action="store_true", help="Use DFS body order instead of BFS")
    parser.add_argument("--body_order_file", type=str, help="Python file defining T800_BODY_ORDER")
    parser.add_argument("--urdf", type=str, default=None, help="Path to T800 URDF; auto-detected by default")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    urdf_path = resolve_urdf(args.urdf)
    body_order = load_body_order(args.body_order_file, args.use_dfs)
    raw_motion = np.load(input_path, allow_pickle=False)
    if raw_motion.ndim == 1:
        raw_motion = np.asarray(raw_motion.tolist(), dtype=np.float32)

    base_pos, base_quat, joint_pos = resample_motion(raw_motion, args.input_fps, args.fps)
    dt = 1.0 / args.fps
    joint_vel = compute_linear_velocity(joint_pos, dt)

    incoming_joints = parse_urdf_joints(urdf_path)
    link_positions, link_quaternions = forward_kinematics(base_pos, base_quat, joint_pos, incoming_joints)
    body_pos, body_quat, body_lin_vel, body_ang_vel = build_body_arrays(body_order, link_positions, link_quaternions, dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_non_zip64(
        output_path,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        fps=np.asarray([args.fps], dtype=np.float32),
    )

    print(f"[npy_to_npz] URDF: {urdf_path}")
    print(f"[npy_to_npz] Input: {input_path}")
    print(f"[npy_to_npz] Output: {output_path}")
    print(f"[npy_to_npz] Frames: {joint_pos.shape[0]}, Joints: {joint_pos.shape[1]}, Bodies: {len(body_order)}")
    print(f"[npy_to_npz] Body order: {'DFS' if args.use_dfs else 'BFS'}" if args.body_order_file is None else "[npy_to_npz] Body order: file")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
