import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


T800_NOKOV_PARENT_MAP = {
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Spine3": "Spine2",
    "LeftShoulder": "Spine3",
    "LeftArm": "Spine3",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine3",
    "RightArm": "Spine3",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFootMod": "LeftLeg",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFootMod": "RightLeg",
}

T800_NOKOV_SEGMENT_LENGTHS = {
    "Spine": 0.02596,
    "Spine1": 0.02596,
    "Spine2": 0.02596,
    "Spine3": 0.02596,
    "LeftShoulder": 0.27889,
    "RightShoulder": 0.27889,
    "LeftArm": 0.32323,
    "RightArm": 0.32323,
    "LeftForeArm": 0.26293,
    "RightForeArm": 0.26293,
    "LeftHand": 0.26635,
    "RightHand": 0.26635,
    "LeftUpLeg": 0.07918,
    "RightUpLeg": 0.07918,
    "LeftLeg": 0.47824,
    "RightLeg": 0.47824,
    "LeftFootMod": 0.43868,
    "RightFootMod": 0.43868,
}


def scale_t800_nokov_skeleton(frame_data):
    scaled = {
        "Hips": [frame_data["Hips"][0].copy(), frame_data["Hips"][1].copy()]
    }
    for child, parent in T800_NOKOV_PARENT_MAP.items():
        if child not in frame_data or parent not in scaled:
            continue
        direction = frame_data[child][0] - frame_data[parent][0]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        scaled[child] = [
            scaled[parent][0] + direction * T800_NOKOV_SEGMENT_LENGTHS[child],
            frame_data[child][1].copy(),
        ]
    return scaled


def load_bvh_file(
    bvh_file,
    format="lafan1",
    align_to_robot=False,
    t800_segment_scale=False,
):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    robot_align_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    robot_align_quat = R.from_matrix(robot_align_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            if align_to_robot:
                orientation = utils.quat_mul(robot_align_quat, orientation)
                position = robot_align_matrix @ position
            result[bone] = [position, orientation]
            
        if format == "lafan1":
            # Add modified foot pose
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
            if t800_segment_scale:
                result.update(scale_t800_nokov_skeleton(result))
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


