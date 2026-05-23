import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


LEFT_ARM_DOF = np.arange(13, 18)  # J13..J17 in dof_pos.


def joint_bounds_for_dof_indices(model, dof_indices):
    lows, highs = [], []
    for dof_index in dof_indices:
        qpos_index = dof_index + 7
        joint_id = np.where(model.jnt_qposadr == qpos_index)[0][0]
        lows.append(model.jnt_range[joint_id, 0])
        highs.append(model.jnt_range[joint_id, 1])
    return np.asarray(lows), np.asarray(highs)


def local_body_pos(model, data, body_ids, root_pos, root_rot, dof_pos, body_name):
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot
    data.qpos[7:] = dof_pos
    mj.mj_forward(model, data)
    base_pos = data.xpos[body_ids["LINK_BASE"]]
    base_mat = data.xmat[body_ids["LINK_BASE"]].reshape(3, 3)
    return base_mat.T @ (data.xpos[body_ids[body_name]] - base_pos)


def fix_initial_left_arm(
    motion,
    bvh_file,
    solve_end_frames,
    blend_end_frames,
):
    frames, actual_human_height = load_bvh_file(
        str(bvh_file),
        format="nokov",
        align_to_robot=True,
    )
    retargeter = GMR(
        src_human="bvh_nokov",
        tgt_robot="t800",
        actual_human_height=actual_human_height,
        verbose=False,
    )

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["t800"]))
    data = mj.MjData(model)
    body_ids = {
        name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        for name in ["LINK_BASE", "LINK_ELBOW_PITCH_L", "LINK_WRIST_END_L"]
    }
    bounds = joint_bounds_for_dof_indices(model, LEFT_ARM_DOF)

    root_pos = np.asarray(motion["root_pos"])
    root_rot = np.asarray(motion["root_rot"])
    dof_pos = np.asarray(motion["dof_pos"]).copy()
    original = dof_pos.copy()

    end = min(blend_end_frames, dof_pos.shape[0], len(frames))
    solve_end = min(solve_end_frames, end)
    previous_arm = dof_pos[0, LEFT_ARM_DOF].copy()

    for frame in range(end):
        retargeter.update_targets(frames[frame], offset_to_ground=True)
        human_data = retargeter.scaled_human_data
        human_root = np.asarray(human_data["Hips"][0])
        target_elbow = np.asarray(human_data["LeftForeArm"][0]) - human_root
        target_wrist = np.asarray(human_data["LeftHand"][0]) - human_root

        if frame < solve_end:
            base_dof = dof_pos[frame].copy()
            original_arm = original[frame, LEFT_ARM_DOF].copy()

            def residual(arm_q):
                candidate = base_dof.copy()
                candidate[LEFT_ARM_DOF] = arm_q
                elbow = local_body_pos(
                    model, data, body_ids, root_pos[frame], root_rot[frame], candidate, "LINK_ELBOW_PITCH_L"
                )
                wrist = local_body_pos(
                    model, data, body_ids, root_pos[frame], root_rot[frame], candidate, "LINK_WRIST_END_L"
                )
                return np.concatenate(
                    [
                        3.0 * (wrist - target_wrist),
                        1.2 * (elbow - target_elbow),
                        0.10 * (arm_q - original_arm),
                        0.25 * (arm_q - previous_arm),
                    ]
                )

            result = least_squares(
                residual,
                np.clip(previous_arm, bounds[0], bounds[1]),
                bounds=bounds,
                max_nfev=80,
                xtol=1e-5,
                ftol=1e-5,
                gtol=1e-5,
            )
            dof_pos[frame, LEFT_ARM_DOF] = result.x
            previous_arm = result.x
        else:
            t = (frame - solve_end) / max(1, end - solve_end)
            alpha = 0.5 * (1.0 + np.cos(np.pi * t))
            dof_pos[frame, LEFT_ARM_DOF] = (
                alpha * previous_arm + (1.0 - alpha) * original[frame, LEFT_ARM_DOF]
            )

    dof_pos[end:, LEFT_ARM_DOF] = original[end:, LEFT_ARM_DOF]
    fixed = dict(motion)
    fixed["dof_pos"] = dof_pos
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bvh_file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solve_end_frames", default=180, type=int)
    parser.add_argument("--blend_end_frames", default=240, type=int)
    args = parser.parse_args()

    with args.input.open("rb") as f:
        motion = pickle.load(f)

    fixed = fix_initial_left_arm(
        motion,
        bvh_file=args.bvh_file,
        solve_end_frames=args.solve_end_frames,
        blend_end_frames=args.blend_end_frames,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(fixed, f)
    print(f"Saved fixed motion to {args.output}")


if __name__ == "__main__":
    main()
