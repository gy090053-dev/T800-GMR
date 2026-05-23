import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import addon_utils


MANUAL_NOKOV_TO_XSENS = {
    "Hips": "Hips",
    "Spine": "Chest",
    "Spine1": "Chest2",
    "Spine2": "Chest3",
    "Spine3": "Chest4",
    "Neck": "Neck",
    "Head": "Head",
    "LeftShoulder": "LeftCollar",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftElbow",
    "LeftHand": "LeftWrist",
    "RightShoulder": "RightCollar",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightElbow",
    "RightHand": "RightWrist",
    "LeftUpLeg": "LeftHip",
    "LeftLeg": "LeftKnee",
    "LeftFoot": "LeftAnkle",
    "LeftToeBase": "LeftToe",
    "RightUpLeg": "RightHip",
    "RightLeg": "RightKnee",
    "RightFoot": "RightAnkle",
    "RightToeBase": "RightToe",
}


def ensure_rokoko_lib_info():
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return
    info_path = Path(appdata) / "Blender Foundation" / "Blender" / "5.1" / "scripts" / "addons" / "Rokoko Libraries" / ".lib_info"
    if info_path.parent.exists() and not info_path.exists():
        info_path.write_text('{"operating_system":"Windows"}', encoding="utf-8")


def install_blender_51_action_fcurves_compat():
    if not hasattr(bpy.types.Action, "fcurves"):
        class FCurvesCompat:
            def __init__(self, inner):
                self._inner = inner

            def __iter__(self):
                return iter(self._inner)

            def __len__(self):
                return len(self._inner)

            def __getitem__(self, index):
                return self._inner[index]

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def new(self, data_path, index=0, action_group=None, group_name=None):
                return self._inner.new(
                    data_path=data_path,
                    index=index,
                    group_name=group_name if group_name is not None else action_group,
                )

        def action_fcurves(action):
            if not action.layers:
                layer = action.layers.new(name="Layer")
            else:
                layer = action.layers[0]

            if not layer.strips:
                strip = layer.strips.new(type="KEYFRAME")
            else:
                strip = layer.strips[0]

            if not action.slots:
                slot = action.slots.new(id_type="OBJECT", name="Slot")
            else:
                slot = action.slots[0]

            if not strip.channelbags:
                channelbag = strip.channelbags.new(slot)
            else:
                channelbag = strip.channelbags[0]

            return FCurvesCompat(channelbag.fcurves)

        bpy.types.Action.fcurves = property(action_fcurves)

    if not hasattr(bpy.types.Bone, "select"):
        def get_bone_select(bone):
            for obj in bpy.data.objects:
                if obj.type == "ARMATURE" and obj.data == bone.id_data and bone.name in obj.pose.bones:
                    return obj.pose.bones[bone.name].select
            return False

        def set_bone_select(bone, value):
            for obj in bpy.data.objects:
                if obj.type == "ARMATURE" and obj.data == bone.id_data and bone.name in obj.pose.bones:
                    obj.pose.bones[bone.name].select = value

        bpy.types.Bone.select = property(get_bone_select, set_bone_select)


def import_bvh(path: str):
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path)
    created = [obj for obj in bpy.data.objects if obj not in before and obj.type == "ARMATURE"]
    if not created:
        raise RuntimeError(f"No armature imported from {path}")
    return created[-1]


def limit_action_frames(armature, max_frames: int):
    action = armature.animation_data.action if armature.animation_data else None
    if not action:
        return
    for fcurve in action.fcurves:
        for key in reversed(fcurve.keyframe_points):
            if key.co.x > max_frames:
                fcurve.keyframe_points.remove(key)
        fcurve.update()


def summarize_armature(armature):
    return {
        "name": armature.name,
        "bone_count": len(armature.pose.bones),
        "bones": [bone.name for bone in armature.pose.bones],
        "has_action": bool(armature.animation_data and armature.animation_data.action),
        "action": armature.animation_data.action.name if armature.animation_data and armature.animation_data.action else None,
    }


def add_manual_mapping(scene, source, target):
    scene.rsl_retargeting_bone_list.clear()
    source_names = set(source.pose.bones.keys())
    target_names = set(target.pose.bones.keys())
    added = []
    skipped = []
    for source_bone, target_bone in MANUAL_NOKOV_TO_XSENS.items():
        if source_bone in source_names and target_bone in target_names:
            item = scene.rsl_retargeting_bone_list.add()
            item.bone_name_source = source_bone
            item.bone_name_target = target_bone
            item.bone_name_key = source_bone
            item.is_custom = True
            added.append({"source": source_bone, "target": target_bone})
        else:
            skipped.append({"source": source_bone, "target": target_bone})
    return added, skipped


def export_target_bvh(target, out_path: str, frame_start: int, frame_end: int):
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.export_anim.bvh(
        filepath=out_path,
        frame_start=frame_start,
        frame_end=frame_end,
        root_transform_only=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out_bvh", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--frames", type=int, default=120)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    install_blender_51_action_fcurves_compat()
    ensure_rokoko_lib_info()
    addon_utils.enable("rokoko-studio-live-blender-master", default_set=True)

    source = import_bvh(args.source)
    source.name = "SOURCE_OFFICIAL_NOKOV_LIKE"
    limit_action_frames(source, args.frames)

    target = import_bvh(args.target)
    target.name = "TARGET_XSENS_SKELETON"
    if target.animation_data:
        target.animation_data.action = None

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.rsl_retargeting_armature_source = source
    scene.rsl_retargeting_armature_target = target
    scene.rsl_retargeting_auto_scaling = True
    scene.rsl_retargeting_use_pose = "REST"

    auto_result = None
    auto_count = 0
    try:
        auto_result = sorted(bpy.ops.rsl.build_bone_list())
        auto_count = len(scene.rsl_retargeting_bone_list)
    except Exception as exc:
        auto_result = [f"ERROR: {type(exc).__name__}: {exc}"]

    manual_added, manual_skipped = add_manual_mapping(scene, source, target)

    retarget_result = sorted(bpy.ops.rsl.retarget_animation())
    target_action = target.animation_data.action if target.animation_data else None
    export_ok = bool(target_action)
    if export_ok:
        export_target_bvh(target, args.out_bvh, 1, args.frames)

    report = {
        "source_file": args.source,
        "target_file": args.target,
        "source": summarize_armature(source),
        "target": summarize_armature(target),
        "auto_build_result": auto_result,
        "auto_mapping_count": auto_count,
        "manual_mapping_count": len(manual_added),
        "manual_mapping": manual_added,
        "manual_skipped": manual_skipped,
        "retarget_result": retarget_result,
        "target_action": target_action.name if target_action else None,
        "exported_bvh": args.out_bvh if export_ok else None,
        "frames": args.frames,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
