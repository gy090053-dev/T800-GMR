import json
import sys
from pathlib import Path

import bpy
import addon_utils


def parse_args():
    args = sys.argv
    if "--" not in args:
        raise SystemExit("Usage: blender --background --python inspect_bvh_blender.py -- <bvh...> --out <report.json>")
    user_args = args[args.index("--") + 1 :]
    out = None
    bvh_files = []
    i = 0
    while i < len(user_args):
        if user_args[i] == "--out":
            out = Path(user_args[i + 1])
            i += 2
        else:
            bvh_files.append(Path(user_args[i]))
            i += 1
    if not out or not bvh_files:
        raise SystemExit("Missing BVH file or --out path")
    return bvh_files, out


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_bvh(path):
    reset_scene()
    addon_utils.enable("io_anim_bvh")
    bpy.ops.import_anim.bvh(
        filepath=str(path),
        global_scale=1.0,
        update_scene_fps=True,
        update_scene_duration=True,
    )
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature imported from {path}")
    return armatures[0]


def find_first(names, candidates):
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def world_position(armature, pose_bone_name):
    pose_bone = armature.pose.bones[pose_bone_name]
    matrix = armature.matrix_world @ pose_bone.matrix
    return [round(v, 6) for v in matrix.translation]


def inspect_bvh(path):
    armature = import_bvh(path)
    bones = armature.data.bones
    bone_names = [bone.name for bone in bones]
    root_bones = [bone.name for bone in bones if bone.parent is None]
    parent_map = {bone.name: bone.parent.name if bone.parent else None for bone in bones}

    key_candidates = {
        "root": ["Hips", "Hip", "Pelvis", "Root", "root"],
        "left_hand": ["LeftHand", "LeftWrist", "LeftHandIndex", "L_Hand", "mixamorig:LeftHand"],
        "right_hand": ["RightHand", "RightWrist", "RightHandIndex", "R_Hand", "mixamorig:RightHand"],
        "left_foot": ["LeftFoot", "LeftToeBase", "LeftAnkle", "L_Foot", "mixamorig:LeftFoot"],
        "right_foot": ["RightFoot", "RightToeBase", "RightAnkle", "R_Foot", "mixamorig:RightFoot"],
        "left_shoulder": ["LeftShoulder", "LeftArm", "L_Shoulder", "mixamorig:LeftShoulder"],
        "right_shoulder": ["RightShoulder", "RightArm", "R_Shoulder", "mixamorig:RightShoulder"],
        "left_elbow": ["LeftForeArm", "LeftElbow", "L_ForeArm", "mixamorig:LeftForeArm"],
        "right_elbow": ["RightForeArm", "RightElbow", "R_ForeArm", "mixamorig:RightForeArm"],
        "left_knee": ["LeftLeg", "LeftKnee", "L_Leg", "mixamorig:LeftLeg"],
        "right_knee": ["RightLeg", "RightKnee", "R_Leg", "mixamorig:RightLeg"],
    }
    key_bones = {
        key: find_first(bone_names, candidates)
        for key, candidates in key_candidates.items()
    }

    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    sample_frames = sorted(set([frame_start, min(frame_start + 100, frame_end), (frame_start + frame_end) // 2, frame_end]))
    samples = {}
    for frame in sample_frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_sample = {}
        for key, bone_name in key_bones.items():
            if bone_name and bone_name in armature.pose.bones:
                frame_sample[key] = {
                    "bone": bone_name,
                    "world_pos": world_position(armature, bone_name),
                }
        samples[str(frame)] = frame_sample

    return {
        "file": str(path),
        "armature": armature.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "fps": scene.render.fps,
        "bone_count": len(bone_names),
        "root_bones": root_bones,
        "key_bones": key_bones,
        "bone_names": bone_names,
        "parent_map": parent_map,
        "samples": samples,
    }


def main():
    bvh_files, out = parse_args()
    reports = [inspect_bvh(path) for path in bvh_files]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote BVH inspection report: {out}")


if __name__ == "__main__":
    main()
