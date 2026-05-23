import argparse
import sys
from pathlib import Path

import bpy


def parse_args():
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender --background --python blender_bvh_roundtrip.py -- --input in.bvh --output out.bvh")
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--axis_forward", default="-Z")
    parser.add_argument("--axis_up", default="Y")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    input_path = str(Path(args.input).resolve())
    output_path = str(Path(args.output).resolve())

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.preferences.addon_enable(module="io_anim_bvh")
    bpy.ops.import_anim.bvh(
        filepath=input_path,
        axis_forward=args.axis_forward,
        axis_up=args.axis_up,
        rotate_mode="NATIVE",
    )

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature imported from BVH")

    armature = armatures[0]
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)

    action = armature.animation_data.action if armature.animation_data else None
    frame_start = int(bpy.context.scene.frame_start)
    frame_end = int(bpy.context.scene.frame_end)
    if action is not None:
        frame_start, frame_end = map(int, action.frame_range)

    bone_names = [bone.name for bone in armature.data.bones]
    print("BVH_IMPORT_OK")
    print(f"armature={armature.name}")
    print(f"bones={len(bone_names)}")
    print(f"frames={frame_start}-{frame_end}")
    print("first_bones=" + ",".join(bone_names[:20]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(
        filepath=output_path,
        frame_start=frame_start,
        frame_end=frame_end,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )
    print(f"BVH_EXPORT_OK {output_path}")


if __name__ == "__main__":
    main()
