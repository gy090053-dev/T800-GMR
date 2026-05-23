# T800 GMR 接入与修改说明

本文档说明在原版 GMR 基础上，为实现官方 BVH 动作重定向到 T800 机器人所做的主要改动。

## 目标

将官方 BVH 动作文件通过 GMR 重定向到 T800 MuJoCo 机器人模型，并导出 T800 可用的 `pkl` 动作数据。

当前主路线为：

```text
官方 BVH
  -> format=nokov
  -> GMR IK
  -> robot=t800
  -> T800 pkl
```

## 1. 接入 T800 机器人模型

在 `general_motion_retargeting/params.py` 中注册了 T800：

```text
ROBOT_XML_DICT["t800"]
ROBOT_BASE_DICT["t800"] = "LINK_BASE"
VIEWER_CAM_DISTANCE_DICT["t800"]
```

T800 MuJoCo 模型路径：

```text
E:\T800\T800_model\xml\serial_t800.xml
```

如果对方把 T800 资产放进 GMR 的 `assets` 目录，需要同步修改 `params.py` 中 `ROBOT_XML_DICT["t800"]` 的路径。

## 2. 新增 T800 IK 配置

新增了 T800 相关 IK 配置文件：

```text
general_motion_retargeting/ik_configs/bvh_nokov_to_t800.json
general_motion_retargeting/ik_configs/bvh_lafan1_to_t800.json
general_motion_retargeting/ik_configs/bvh_xsens_to_t800.json
```

当前主要使用：

```text
general_motion_retargeting/ik_configs/bvh_nokov_to_t800.json
```

该文件定义了 BVH 人体骨骼与 T800 link 的映射关系，例如：

```text
Hips       -> LINK_BASE
Spine2     -> LINK_WAIST_YAW
LeftArm    -> LINK_SHOULDER_PITCH_L
LeftForeArm -> LINK_ELBOW_PITCH_L
LeftHand   -> LINK_WRIST_END_L
LeftFoot   -> LINK_ANKLE_PITCH_L
```

同时在该配置中调整了：

```text
human_scale_table
IK 位置权重
default_joint_pose
posture_cost
脚底贴地相关设置
```

## 3. 确认官方 BVH 更接近 Nokov 类

使用 Blender 导入官方 BVH，检查骨架层级和骨骼命名，确认其结构更接近 Nokov 类 BVH。

关键骨骼名如下：

```text
Root / 骨盆: Hips
左手: LeftHand
右手: RightHand
左脚: LeftFoot
右脚: RightFoot
左臂: LeftArm
右臂: RightArm
左前臂: LeftForeArm
右前臂: RightForeArm
左腿: LeftLeg
右腿: RightLeg
```

因此当前推荐使用：

```text
--format nokov --robot t800
```

不建议盲目切换到 `lafan1` 或 `xsens`，除非 BVH 源文件确实是对应骨架。

## 4. 修改 BVH 读取与坐标系对齐

修改了：

```text
general_motion_retargeting/utils/lafan1.py
```

虽然文件名叫 `lafan1.py`，但这里实际承担通用 BVH 读取工作。新增了：

```text
load_bvh_file(..., align_to_robot=True)
```

用于将官方 BVH 坐标系对齐到 T800 / MuJoCo 坐标系。

对于：

```text
--format nokov --robot t800
```

脚本默认启用该坐标系对齐。

## 5. 修复 BVH 多旋转通道解析

修改了：

```text
general_motion_retargeting/utils/lafan_vendor/extract.py
```

原因是官方 BVH 中不同关节的 `CHANNELS` 旋转顺序不完全一致，例如：

```text
Zrotation Yrotation Xrotation
Yrotation Zrotation Xrotation
Xrotation Yrotation Zrotation
```

原始解析逻辑更偏向假设统一旋转顺序，容易导致某些关节姿态解析错误。

现在改为按每个 joint 自己的 `CHANNELS` 顺序解析旋转。

## 6. 修改 BVH 到机器人脚本

修改了：

```text
scripts/bvh_to_robot.py
```

主要新增：

```text
--robot t800
--no_viewer
--align_to_robot_frame
--no_align_to_robot_frame
--offset_to_ground
--no_offset_to_ground
--initial_warmup_iters
```

对于 T800 + Nokov 路线，默认行为为：

```text
align_to_robot_frame = True
offset_to_ground = True
initial_warmup_iters = 0
```

同时修正了保存逻辑，确保导出的 `pkl` 包含完整帧数据。

## 7. 增加 IK 姿态约束能力

修改了：

```text
general_motion_retargeting/motion_retarget.py
```

新增读取 IK 配置中的：

```text
posture_cost
default_joint_pose
```

用于给 T800 一个较自然的默认姿态约束，减少 IK 求解时出现极端扭曲姿态的概率。

## 8. 起始左臂局部修正

官方 BVH 重定向到 T800 后，起始阶段左臂容易进入错误 IK 分支。

新增脚本：

```text
scripts/fix_t800_initial_left_arm.py
```

该脚本不会全局改变动作，而是：

```text
根据 BVH 中 LeftHand / LeftForeArm 的目标位置
只重算起始阶段左臂
随后平滑过渡回原始动作
后续动作保持不变
```

这是一个针对当前官方拳击 BVH 的局部后处理，不属于通用 BVH 解析逻辑。

## 9. Blender 辅助检查脚本

新增脚本：

```text
scripts/inspect_bvh_blender.py
```

用于调用 Blender 后台导入 BVH，并导出骨架检查报告。

主要用途：

```text
检查 BVH 是否能被 Blender 正常导入
查看 BVH 骨骼层级
确认 Root / 手 / 脚 / 肩 / 肘 / 膝 等关键骨骼名称
判断 BVH 更接近哪一类骨架
```

## 10. 运行命令

进入 GMR 目录：

```powershell
cd E:\T800\GMR
```

导出 T800 pkl：

```powershell
python .\scripts\bvh_to_robot.py --bvh_file "E:\T800\date\zhiquan_quanji_001.bvh" --format nokov --robot t800 --motion_fps 120 --save_path .\retargeting_data\t800_zhiquan_quanji_001.pkl --no_viewer
```

查看 T800 pkl 动画：

```powershell
python .\scripts\vis_robot_motion.py --robot t800 --robot_motion_path ".\retargeting_data\t800_zhiquan_quanji_001.pkl"
```

如果需要对当前拳击动作修正起始左臂：

```powershell
python .\scripts\fix_t800_initial_left_arm.py --input .\retargeting_data\t800_zhiquan_quanji_001_before_left_start_fix.pkl --bvh_file "E:\T800\date\zhiquan_quanji_001.bvh" --output .\retargeting_data\t800_zhiquan_quanji_001.pkl
```

## 11. 交付给别人时需要包含的内容

如果对方已经有同一版代码，只需要给：

```text
general_motion_retargeting/ik_configs/bvh_nokov_to_t800.json
```

如果对方是原版 GMR，需要同时给：

```text
T800_model/
general_motion_retargeting/params.py
general_motion_retargeting/motion_retarget.py
general_motion_retargeting/utils/lafan1.py
general_motion_retargeting/utils/lafan_vendor/extract.py
scripts/bvh_to_robot.py
scripts/fix_t800_initial_left_arm.py
scripts/inspect_bvh_blender.py
general_motion_retargeting/ik_configs/bvh_nokov_to_t800.json
```

## 12. 当前结论

Blender 检查显示官方 BVH 的骨架结构更接近 Nokov 类，因此当前采用 Nokov 路线是合理的。

当前主要工作不是切换 LAFAN1 / Xsens，而是继续微调：

```text
general_motion_retargeting/ik_configs/bvh_nokov_to_t800.json
```

也就是继续优化 T800 的 IK 映射关系、权重、比例和默认姿态。
