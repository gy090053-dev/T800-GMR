import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def limit_root_tilt(
    qpos,
    roll_limit_deg,
    backward_pitch_limit_deg,
    forward_pitch_limit_deg,
):
    limited_qpos = np.asarray(qpos).copy()
    roll_limit = np.deg2rad(roll_limit_deg)
    backward_pitch_limit = np.deg2rad(backward_pitch_limit_deg)
    forward_pitch_limit = np.deg2rad(forward_pitch_limit_deg)

    root_euler = R.from_quat(limited_qpos[3:7], scalar_first=True).as_euler("xyz")
    root_euler[0] = np.clip(root_euler[0], -roll_limit, roll_limit)
    root_euler[1] = np.clip(root_euler[1], -backward_pitch_limit, forward_pitch_limit)
    limited_qpos[3:7] = R.from_euler("xyz", root_euler).as_quat(scalar_first=True)
    return limited_qpos


def update_solver_state(retargeter, qpos):
    retargeter.configuration.update(qpos)


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "t800"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="Run retargeting without opening the MuJoCo viewer.",
    )

    parser.add_argument(
        "--align_to_robot_frame",
        dest="align_to_robot_frame",
        action="store_true",
        default=None,
        help="Rotate BVH coordinates to MuJoCo robot frame before retargeting.",
    )

    parser.add_argument(
        "--no_align_to_robot_frame",
        dest="align_to_robot_frame",
        action="store_false",
        help="Disable automatic BVH-to-robot coordinate alignment.",
    )

    parser.add_argument(
        "--offset_to_ground",
        dest="offset_to_ground",
        action="store_true",
        default=None,
        help="Shift the human motion so the lowest foot stays near the ground.",
    )

    parser.add_argument(
        "--no_offset_to_ground",
        dest="offset_to_ground",
        action="store_false",
        help="Disable ground-relative shifting of the human motion.",
    )

    parser.add_argument(
        "--initial_warmup_iters",
        default=None,
        type=int,
        help="Solve the first frame this many extra times before playback/export.",
    )

    parser.add_argument(
        "--t800_segment_scale",
        dest="t800_segment_scale",
        action="store_true",
        default=None,
        help="Rescale Nokov BVH body segments to T800 link lengths before IK.",
    )

    parser.add_argument(
        "--no_t800_segment_scale",
        dest="t800_segment_scale",
        action="store_false",
        help="Disable T800-specific Nokov BVH segment scaling.",
    )

    parser.add_argument(
        "--t800_root_level",
        dest="t800_root_level",
        action="store_true",
        default=None,
        help="Limit T800 floating-base roll/pitch drift while preserving heading.",
    )

    parser.add_argument(
        "--no_t800_root_level",
        dest="t800_root_level",
        action="store_false",
        help="Disable T800 floating-base roll/pitch limiting.",
    )

    parser.add_argument(
        "--t800_root_roll_limit_deg",
        default=5.0,
        type=float,
        help="Maximum absolute T800 root roll angle when root leveling is enabled.",
    )

    parser.add_argument(
        "--t800_root_backward_pitch_limit_deg",
        default=4.0,
        type=float,
        help="Maximum T800 root backward pitch angle when root leveling is enabled.",
    )

    parser.add_argument(
        "--t800_root_forward_pitch_limit_deg",
        default=12.0,
        type=float,
        help="Maximum T800 root forward pitch angle when root leveling is enabled.",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )
    
    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    if args.align_to_robot_frame is None:
        align_to_robot_frame = args.robot == "t800" and args.format == "nokov"
    else:
        align_to_robot_frame = args.align_to_robot_frame

    if args.offset_to_ground is None:
        offset_to_ground = args.robot == "t800" and args.format == "nokov"
    else:
        offset_to_ground = args.offset_to_ground

    if args.initial_warmup_iters is None:
        initial_warmup_iters = 15 if args.robot == "t800" and args.format == "nokov" else 0
    else:
        initial_warmup_iters = args.initial_warmup_iters

    if args.t800_segment_scale is None:
        use_t800_segment_scale = args.robot == "t800" and args.format == "nokov"
    else:
        use_t800_segment_scale = args.t800_segment_scale

    if args.t800_root_level is None:
        use_t800_root_level = args.robot == "t800" and args.format == "nokov"
    else:
        use_t800_root_level = args.t800_root_level

    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(
        args.bvh_file,
        format=args.format,
        align_to_robot=align_to_robot_frame,
        t800_segment_scale=use_t800_segment_scale,
    )
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    for _ in range(initial_warmup_iters):
        retargeter.retarget(lafan1_data_frames[0], offset_to_ground=offset_to_ground)

    motion_fps = args.motion_fps
    
    robot_motion_viewer = None
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                motion_fps=motion_fps,
                                                transparent_robot=0,
                                                record_video=args.record_video,
                                                video_path=args.video_path,
                                                # video_width=2080,
                                                # video_height=1170
                                                )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data, offset_to_ground=offset_to_ground)

        if use_t800_root_level:
            qpos = limit_root_tilt(
                qpos,
                args.t800_root_roll_limit_deg,
                args.t800_root_backward_pitch_limit_deg,
                args.t800_root_forward_pitch_limit_deg,
            )
            update_solver_state(retargeter, qpos)

        if args.save_path is not None:
            qpos_list.append(qpos)

        # visualize
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
       
