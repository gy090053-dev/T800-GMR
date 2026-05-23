import argparse
import copy
import sys
from pathlib import Path

import bpy


ROKOKO_ADDON = "rokoko-studio-live-blender-master"


def parse_args():
    if "--" not in sys.argv:
        raise SystemExit(
            "Usage: blender --background --python blender_rokoko_retarget_to_xsens.py -- "
            "--source source.bvh --target_template xsens.bvh --output out.bvh"
        )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target_template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--axis_forward", default="Y")
    parser.add_argument("--axis_up", default="Z")
    parser.add_argument("--source_axis_forward", default=None)
    parser.add_argument("--source_axis_up", default=None)
    parser.add_argument("--target_axis_forward", default=None)
    parser.add_argument("--target_axis_up", default=None)
    parser.add_argument("--use_pose", choices=["REST", "CURRENT"], default="REST")
    parser.add_argument("--auto_scaling", action="store_true", default=True)
    parser.add_argument("--no_auto_scaling", dest="auto_scaling", action="store_false")
    return parser.parse_args(argv)


def enable_addons():
    bpy.ops.preferences.addon_enable(module="io_anim_bvh")

    import addon_utils

    addon = addon_utils.enable(ROKOKO_ADDON, default_set=False, persistent=False)
    if addon is None:
        raise RuntimeError(f"Rokoko addon not found: {ROKOKO_ADDON}")

    # Avoid writing custom retarget schemes into the add-on folder during headless tests.
    for name, module in list(sys.modules.items()):
        if name.endswith(".core.custom_schemes_manager"):
            module.save_retargeting_to_list = lambda: None

    patch_rokoko_for_blender_51()


def read_bvh_frame_time(path):
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.lower().startswith("frame time:"):
                return float(stripped.split(":", 1)[1].strip())
    return None


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def action_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)

    curves = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                curves.extend(list(channelbag.fcurves))
    return curves


def import_bvh(path, name, axis_forward, axis_up):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(
        filepath=str(path),
        axis_forward=axis_forward,
        axis_up=axis_up,
        rotate_mode="NATIVE",
    )
    after = set(bpy.context.scene.objects)
    imported = [obj for obj in after - before if obj.type == "ARMATURE"]
    if not imported:
        imported = [obj for obj in bpy.context.selected_objects if obj.type == "ARMATURE"]
    if not imported:
        raise RuntimeError(f"No armature imported from {path}")

    armature = imported[0]
    armature.name = name
    armature.data.name = name + "_data"
    return armature


def frame_range(armature):
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return int(bpy.context.scene.frame_start), int(bpy.context.scene.frame_end)
    start, end = action.frame_range
    return int(start), int(end)


def print_bones(label, armature, count=25):
    names = [bone.name for bone in armature.data.bones]
    print(f"{label}_BONES={len(names)}")
    print(f"{label}_FIRST_BONES=" + ",".join(names[:count]))


def find_loaded_module(suffix):
    for name, module in sys.modules.items():
        if name.endswith(suffix):
            return module
    raise RuntimeError(f"Module not loaded: {suffix}")


def patch_rokoko_for_blender_51():
    retargeting_ops = find_loaded_module(".operators.retargeting")
    retarget_id = retargeting_ops.RETARGET_ID
    utils = retargeting_ops.utils
    custom_schemes_manager = retargeting_ops.custom_schemes_manager
    get_source_armature = retargeting_ops.get_source_armature
    get_target_armature = retargeting_ops.get_target_armature

    def clean_animation(self, armature_source):
        # Blender 5.x stores f-curves under layered action channelbags. BVH imports
        # put motion on pose bones, so object transform cleanup is unnecessary here.
        return None

    def read_anim_start_end(self, armature):
        action = armature.animation_data.action if armature.animation_data else None
        if action is None:
            return int(bpy.context.scene.frame_start), int(bpy.context.scene.frame_end)

        frame_start = None
        frame_end = None
        for fcurve in action_fcurves(action):
            for key in fcurve.keyframe_points:
                keyframe = key.co.x
                if frame_start is None or keyframe < frame_start:
                    frame_start = keyframe
                if frame_end is None or keyframe > frame_end:
                    frame_end = keyframe

        if frame_start is None or frame_end is None:
            return int(bpy.context.scene.frame_start), int(bpy.context.scene.frame_end)
        return int(frame_start), int(frame_end)

    def bake_animation(self, armature_source, armature_target, root_bones):
        frame_start, frame_end = self.read_anim_start_end(armature_source)
        bpy.ops.object.select_all(action="DESELECT")
        armature_target.select_set(True)
        bpy.context.view_layer.objects.active = armature_target
        bpy.ops.object.mode_set(mode="POSE")
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            visual_keying=True,
            only_selected=True,
            use_current_action=False,
            bake_types={"POSE"},
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        action = armature_target.animation_data.action if armature_target.animation_data else None
        if action is not None:
            action.name = "RSL_RETARGETING_FINAL"
            if hasattr(armature_target.animation_data, "action_slot"):
                suitable_slots = getattr(armature_target.animation_data, "action_suitable_slots", [])
                if suitable_slots:
                    armature_target.animation_data.action_slot = suitable_slots[0]

    def execute(self, context):
        armature_source = get_source_armature()
        armature_target = get_target_armature()

        if not armature_source.animation_data or not armature_source.animation_data.action:
            self.report(
                {"ERROR"},
                "No animation on the source armature found!\n"
                "Select an armature with an animation as source.",
            )
            return {"CANCELLED"}

        if armature_source.name == armature_target.name:
            self.report(
                {"ERROR"},
                "Source and target armature are the same!\nPlease select different armatures.",
            )
            return {"CANCELLED"}

        self.retarget_bone_list.clear()
        for item in context.scene.rsl_retargeting_bone_list:
            if (
                not item.bone_name_source
                or not item.bone_name_target
                or not armature_source.pose.bones.get(item.bone_name_source)
                or not armature_target.pose.bones.get(item.bone_name_target)
            ):
                continue
            self.retarget_bone_list.append(item)

        root_bones = self.find_root_bones(context, armature_source, armature_target)
        if not root_bones:
            self.report(
                {"ERROR"},
                "No root bone found!\n"
                "Check if the bones are mapped correctly or try rebuilding the bone list.",
            )
            return {"CANCELLED"}

        seen = {}
        for item in self.retarget_bone_list:
            count = seen.get(item.bone_name_target)
            if not count:
                count = 0
            seen[item.bone_name_target] = count + 1
        duplicates = [key for key, value in seen.items() if value > 1]
        if duplicates:
            self.report(
                {"ERROR"},
                "Duplicate target bone entries found! Please use each target bone only once:"
                f'\n{", ".join(duplicates)}',
            )
            return {"CANCELLED"}

        custom_schemes_manager.save_retargeting_to_list()

        utils.set_active(armature_target)
        bpy.ops.object.mode_set(mode="OBJECT")
        utils.set_active(armature_source)
        bpy.ops.object.mode_set(mode="OBJECT")

        armature_source.data.pose_position = "POSE"
        armature_target.data.pose_position = "POSE"

        if bpy.context.scene.rsl_retargeting_use_pose == "REST":
            self.get_and_reset_pose_rotations(armature_source)
            self.get_and_reset_pose_rotations(armature_target)

        source_scale = None
        if context.scene.rsl_retargeting_auto_scaling:
            self.clean_animation(armature_source)
            source_scale = copy.deepcopy(armature_source.scale)
            self.scale_armature(context, armature_source, armature_target, root_bones)

        armature_source_original = armature_source
        armature_source = self.copy_rest_pose(context, armature_source)

        rotation_mode = armature_target.rotation_mode
        armature_target.rotation_mode = "QUATERNION"
        rotation = copy.deepcopy(armature_target.rotation_quaternion)
        location = copy.deepcopy(armature_target.location)

        bpy.ops.object.select_all(action="DESELECT")
        utils.set_active(armature_target)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        bpy.ops.object.mode_set(mode="EDIT")

        bone_transforms = {}
        for bone in context.object.data.edit_bones:
            bone.select = False
            bone_transforms[bone.name] = (
                armature_source.matrix_world.inverted() @ bone.head.copy(),
                armature_source.matrix_world.inverted() @ bone.tail.copy(),
                utils.mat3_to_vec_roll(
                    armature_source.matrix_world.inverted().to_3x3() @ bone.matrix.to_3x3()
                ),
            )

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        utils.set_active(armature_source)
        bpy.ops.object.mode_set(mode="EDIT")

        for item in self.retarget_bone_list:
            bone_source = armature_source.data.edit_bones.get(item.bone_name_source)

            bone_new = armature_source.data.edit_bones.new(item.bone_name_target + retarget_id)
            bone_new.head, bone_new.tail, bone_new.roll = bone_transforms[item.bone_name_target]
            bone_new.parent = bone_source

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        utils.set_active(armature_target)
        bpy.ops.object.mode_set(mode="POSE")
        for pose_bone in armature_target.pose.bones:
            pose_bone.select = False

        for item in self.retarget_bone_list:
            bone_target = armature_target.pose.bones.get(item.bone_name_target)

            constraint = bone_target.constraints.new("COPY_ROTATION")
            constraint.name += retarget_id
            constraint.target = armature_source
            constraint.subtarget = item.bone_name_target + retarget_id

            if bone_target.name in root_bones:
                constraint = bone_target.constraints.new("COPY_LOCATION")
                constraint.name += retarget_id
                constraint.target = armature_source
                constraint.subtarget = item.bone_name_source

            bone_target.select = True

        bpy.ops.object.mode_set(mode="OBJECT")
        self.bake_animation(armature_source, armature_target, root_bones)

        bpy.ops.object.select_all(action="DESELECT")
        utils.set_active(armature_source)
        bpy.data.actions.remove(armature_source.animation_data.action)
        bpy.ops.object.delete()

        armature_source = armature_source_original

        armature_target.animation_data.action.name = armature_source.animation_data.action.name + " Retarget"

        for bone in armature_target.pose.bones:
            for constraint in list(bone.constraints):
                if retarget_id in constraint.name:
                    bone.constraints.remove(constraint)

        bpy.ops.object.select_all(action="DESELECT")
        utils.set_active(armature_target)

        armature_target.rotation_quaternion = rotation
        armature_target.location = location

        armature_target.rotation_quaternion.w = -armature_target.rotation_quaternion.w
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        armature_target.rotation_quaternion = rotation
        armature_target.rotation_mode = rotation_mode

        if source_scale:
            armature_source.scale = source_scale

        bpy.ops.object.select_all(action="DESELECT")

        self.report({"INFO"}, "Retargeted animation.")
        return {"FINISHED"}

    retargeting_ops.RetargetAnimation.execute = execute
    retargeting_ops.RetargetAnimation.clean_animation = clean_animation
    retargeting_ops.RetargetAnimation.read_anim_start_end = read_anim_start_end
    retargeting_ops.RetargetAnimation.bake_animation = bake_animation


def build_bone_list(source, target, use_pose, auto_scaling):
    detection_manager = find_loaded_module(".core.detection_manager")

    if not getattr(detection_manager, "bone_detection_list", None):
        detection_manager.load_detection_lists()

    scene = bpy.context.scene
    scene.rsl_retargeting_armature_source = source
    scene.rsl_retargeting_armature_target = target
    scene.rsl_retargeting_use_pose = use_pose
    scene.rsl_retargeting_auto_scaling = auto_scaling

    action = source.animation_data.action if source.animation_data else None
    if action is None:
        raise RuntimeError("Source armature has no action")

    bone_list_animated = []
    for fc in action_fcurves(action):
        bone_name = fc.data_path.split('"')
        if len(bone_name) == 3 and bone_name[1] not in bone_list_animated:
            bone_list_animated.append(bone_name[1])

    retargeting_dict = {}
    found_main_bones = []
    spines_source = []

    for bone_name in bone_list_animated:
        bone_item_source = bone_name
        bone_item_target = ""
        main_bone_name = ""
        standardized_bone_name_source = detection_manager.standardize_bone_name(bone_name)

        for bone_main, bone_values in detection_manager.bone_detection_list.items():
            if bone_main == "chest":
                continue
            if bone_main in found_main_bones:
                continue
            if (
                bone_name.lower() in bone_values
                or standardized_bone_name_source in bone_values
                or standardized_bone_name_source == bone_main.lower()
            ):
                main_bone_name = bone_main
                if main_bone_name != "spine":
                    found_main_bones.append(main_bone_name)
                    break

        retargeting_dict[bone_item_source] = ("", main_bone_name)

        if not main_bone_name:
            continue

        if main_bone_name == "spine":
            spines_source.append(bone_name)
            continue

        custom_main_bone = main_bone_name.startswith("custom_bone_")
        if custom_main_bone and detection_manager.standardize_bone_name(
            main_bone_name.replace("custom_bone_", "")
        ) in detection_manager.bone_detection_list["spine"]:
            spines_source.append(bone_name)

        bone_item_target = detection_manager.detect_bone(
            target, main_bone_name, bone_name_source=bone_item_source
        )
        retargeting_dict[bone_item_source] = (bone_item_target, main_bone_name)

    spines_target = []
    for bone in target.pose.bones:
        bone_name_standardized = detection_manager.standardize_bone_name(bone.name)
        if bone_name_standardized in detection_manager.bone_detection_list["spine"]:
            spines_target.append(bone.name)

    if spines_target and spines_source:
        spine_dict = {}

        i = 0
        for spine in reversed(spines_source):
            i += 1
            if i == len(spines_target):
                break
            spine_dict[spine] = spines_target[-i]

        spine_dict[spines_source[0]] = spines_target[0]

        for spine_source, spine_target in spine_dict.items():
            for bone_source, bone_values in retargeting_dict.items():
                bone_target, bone_key = bone_values
                if bone_source == spine_source and not bone_target:
                    retargeting_dict[bone_source] = (spine_target, bone_key)
                    break

    scene.rsl_retargeting_bone_list.clear()
    for bone_source, bone_values in retargeting_dict.items():
        bone_target, bone_key = bone_values
        bone_item = scene.rsl_retargeting_bone_list.add()
        bone_item.bone_name_key = bone_key
        bone_item.bone_name_source = bone_source
        bone_item.bone_name_target = bone_target

    mappings = list(scene.rsl_retargeting_bone_list)
    mapped = [item for item in mappings if item.bone_name_source and item.bone_name_target]
    missing = [item for item in mappings if item.bone_name_source and not item.bone_name_target]
    print(f"RSL_MAPPING_TOTAL={len(mappings)}")
    print(f"RSL_MAPPING_MAPPED={len(mapped)}")
    print(f"RSL_MAPPING_MISSING={len(missing)}")
    if missing:
        print("RSL_MISSING_SOURCE_BONES=" + ",".join(item.bone_name_source for item in missing[:40]))
    print(
        "RSL_MAPPED_PREVIEW="
        + ",".join(f"{item.bone_name_source}->{item.bone_name_target}" for item in mapped[:40])
    )
    if len(mapped) < 8:
        raise RuntimeError("Too few Rokoko bone mappings; retarget result would be invalid.")


def export_target(target, output_path, start, end):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.export_anim.bvh(
        filepath=str(output_path),
        frame_start=start,
        frame_end=end,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )


def main():
    args = parse_args()
    source_path = Path(args.source).resolve()
    target_template_path = Path(args.target_template).resolve()
    output_path = Path(args.output).resolve()

    enable_addons()
    clear_scene()

    source_frame_time = read_bvh_frame_time(source_path)
    if source_frame_time:
        source_fps = max(1, int(round(1.0 / source_frame_time)))
        bpy.context.scene.render.fps = source_fps
        bpy.context.scene.render.fps_base = 1.0
    print(f"SCENE_FPS={source_fps}")
    print(f"SOURCE_FRAME_TIME={source_frame_time}")

    source_axis_forward = args.source_axis_forward or args.axis_forward
    source_axis_up = args.source_axis_up or args.axis_up
    target_axis_forward = args.target_axis_forward or args.axis_forward
    target_axis_up = args.target_axis_up or args.axis_up

    print(f"SOURCE_AXIS={source_axis_forward},{source_axis_up}")
    print(f"TARGET_AXIS={target_axis_forward},{target_axis_up}")

    source = import_bvh(source_path, "SOURCE_BVH", source_axis_forward, source_axis_up)
    target = import_bvh(target_template_path, "XSENS_TARGET", target_axis_forward, target_axis_up)
    if target.animation_data:
        target.animation_data_clear()

    start, end = frame_range(source)
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end

    print("SOURCE_FILE=", source_path)
    print("TARGET_TEMPLATE=", target_template_path)
    print("OUTPUT_FILE=", output_path)
    print(f"SOURCE_FRAMES={start}-{end}")
    print_bones("SOURCE", source)
    print_bones("TARGET", target)

    build_bone_list(source, target, args.use_pose, args.auto_scaling)

    result = bpy.ops.rsl.retarget_animation()
    print("RSL_RETARGET_ANIMATION=", result)
    if "FINISHED" not in result:
        raise RuntimeError(f"Rokoko retarget failed: {result}")

    export_target(target, output_path, start, end)
    print("BVH_RETARGET_EXPORT_OK", output_path)


if __name__ == "__main__":
    main()
