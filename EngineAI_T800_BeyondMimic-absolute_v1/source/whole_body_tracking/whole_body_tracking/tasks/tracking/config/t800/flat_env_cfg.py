from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg  # This is altered for ending hold

from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG, T800_END_JOINT_POS, T800_MOTION_JOINT_NAMES
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
import whole_body_tracking.tasks.tracking.mdp as mdp  # This is altered for ending hold
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

T800_RIOT_PUSH_VELOCITY_RANGE = {  # This is altered for ending hold
    "x": (-0.5, 0.5),  # This is altered for ending hold
    "y": (-0.5, 0.5),  # This is altered for ending hold
    "z": (-0.2, 0.2),  # This is altered for ending hold
    "roll": (-0.52, 0.52),  # This is altered for ending hold
    "pitch": (-0.52, 0.52),  # This is altered for ending hold
    "yaw": (-0.78, 0.78),  # This is altered for ending hold
}  # This is altered for ending hold

T800_TERMINAL_PUSH_VELOCITY_RANGE = {  # This is altered for ending hold
    "x": (-0.75, 0.75),  # This is altered for ending hold
    "y": (-0.75, 0.75),  # This is altered for ending hold
    "z": (-0.3, 0.3),  # This is altered for ending hold
    "roll": (-0.78, 0.78),  # This is altered for ending hold
    "pitch": (-0.78, 0.78),  # This is altered for ending hold
    "yaw": (-1.17, 1.17),  # This is altered for ending hold
}  # This is altered for ending hold

T800_TERMINAL_JOINT_WEIGHTS = {  # This is altered for ending hold
    "J00_HIP_PITCH_L": 0.25,  # This is altered for ending hold
    "J01_HIP_ROLL_L": 0.25,  # This is altered for ending hold
    "J02_HIP_YAW_L": 0.25,  # This is altered for ending hold
    "J03_KNEE_PITCH_L": 0.35,  # This is altered for ending hold
    "J06_HIP_PITCH_R": 0.25,  # This is altered for ending hold
    "J07_HIP_ROLL_R": 0.25,  # This is altered for ending hold
    "J08_HIP_YAW_R": 0.25,  # This is altered for ending hold
    "J09_KNEE_PITCH_R": 0.35,  # This is altered for ending hold
}  # This is altered for ending hold

T800_ANKLE_JOINT_NAMES = [  # This is altered for ending hold
    "J04_ANKLE_PITCH_L",  # This is altered for ending hold
    "J05_ANKLE_ROLL_L",  # This is altered for ending hold
    "J10_ANKLE_PITCH_R",  # This is altered for ending hold
    "J11_ANKLE_ROLL_R",  # This is altered for ending hold
]  # This is altered for ending hold


@configclass
class T800FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T800_ACTION_SCALE
        self.commands.motion.motion_joint_names = T800_MOTION_JOINT_NAMES
        self.commands.motion.anchor_body_name = "LINK_TORSO_YAW"
        self.commands.motion.body_names = [
            "LINK_BASE",
            "LINK_HIP_ROLL_L",
            "LINK_KNEE_PITCH_L",
            "LINK_ANKLE_ROLL_L",
            "LINK_HIP_ROLL_R",
            "LINK_KNEE_PITCH_R",
            "LINK_ANKLE_ROLL_R",
            "LINK_TORSO_YAW",
            "LINK_SHOULDER_ROLL_L",
            "LINK_ELBOW_PITCH_L",
            "LINK_ELBOW_YAW_L",
            "LINK_SHOULDER_ROLL_R",
            "LINK_ELBOW_PITCH_R",
            "LINK_ELBOW_YAW_R",
            "LINK_HEAD_YAW",
        ]


@configclass
class T800FlatWoStateEstimationEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass  # This is altered for ending hold
class T800FlatWoStateEstimationTerminalHoldEnvCfg(T800FlatWoStateEstimationEnvCfg):  # This is altered for ending hold
    def __post_init__(self):  # This is altered for ending hold
        super().__post_init__()  # This is altered for ending hold
        terminal_body_names = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]  # This is altered for ending hold
        terminal_contact_cfg = SceneEntityCfg("contact_forces", body_names=terminal_body_names)  # This is altered for ending hold
        self.commands.motion.sampling_mode = "sequential"  # This is altered for ending hold
        self.commands.motion.end_behavior = "hold"  # This is altered for ending hold
        self.commands.motion.terminal_quiet_time_range_s = (1.5, 3.0)  # This is altered for ending hold
        self.episode_length_s = 10.0  # This is altered for ending hold
        self.events.push_robot = EventTerm(  # This is altered for ending hold
            func=mdp.push_robot_before_final_frame,  # This is altered for ending hold
            mode="interval",  # This is altered for ending hold
            interval_range_s=(1.0, 3.0),  # This is altered for ending hold
            params={  # This is altered for ending hold
                "command_name": "motion",  # This is altered for ending hold
                "velocity_range": T800_RIOT_PUSH_VELOCITY_RANGE,  # This is altered for ending hold
                "asset_cfg": SceneEntityCfg("robot"),  # This is altered for ending hold
            },  # This is altered for ending hold
        )  # This is altered for ending hold
        self.events.final_push_robot = EventTerm(  # This is altered for ending hold
            func=mdp.push_robot_after_final_frame,  # This is altered for ending hold
            mode="interval",  # This is altered for ending hold
            interval_range_s=(0.5, 1.0),  # This is altered for ending hold
            params={  # This is altered for ending hold
                "command_name": "motion",  # This is altered for ending hold
                "velocity_range": T800_TERMINAL_PUSH_VELOCITY_RANGE,  # This is altered for ending hold
                "push_probability": 0.7,  # This is altered for ending hold
                "hold_grace_time_s": 0.0,  # This is altered for ending hold
                "asset_cfg": SceneEntityCfg("robot"),  # This is altered for ending hold
            },  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_joint_pos = RewTerm(  # This is altered for ending hold
            func=mdp.final_joint_position_deadband_error_exp,  # This is altered for ending hold
            weight=1.0,  # This is altered for ending hold
            params={  # This is altered for ending hold
                "command_name": "motion",  # This is altered for ending hold
                "std": 0.25,  # This is altered for ending hold
                "target_joint_pos": T800_END_JOINT_POS,  # This is altered for ending hold
                "tolerance": 0.1,  # This is altered for ending hold
                "joint_weights": T800_TERMINAL_JOINT_WEIGHTS,  # This is altered for ending hold
            },  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_anchor_ori = RewTerm(  # This is altered for ending hold
            func=mdp.final_anchor_orientation_error_exp,  # This is altered for ending hold
            weight=1.0,  # This is altered for ending hold
            params={"command_name": "motion", "std": 0.35},  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_quiet_base_velocity = RewTerm(  # This is altered for ending hold
            func=mdp.final_quiet_base_velocity_l2,  # This is altered for ending hold
            weight=-1.0,  # This is altered for ending hold
            params={"command_name": "motion", "lin_weight": 1.0, "ang_weight": 0.25},  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_recovery_base_velocity = RewTerm(  # This is altered for ending hold
            func=mdp.final_recovery_base_velocity_l2,  # This is altered for ending hold
            weight=-0.3,  # This is altered for ending hold
            params={"command_name": "motion", "lin_weight": 1.0, "ang_weight": 0.25},  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_quiet_feet_slip = RewTerm(  # This is altered for ending hold
            func=mdp.final_quiet_feet_slip_l2,  # This is altered for ending hold
            weight=-0.2,  # This is altered for ending hold
            params={  # This is altered for ending hold
                "command_name": "motion",  # This is altered for ending hold
                "sensor_cfg": terminal_contact_cfg,  # This is altered for ending hold
                "body_names": terminal_body_names,  # This is altered for ending hold
                "contact_threshold": 0.0,  # This is altered for ending hold
            },  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_recovery_feet_slip = RewTerm(  # This is altered for ending hold
            func=mdp.final_recovery_feet_slip_l2,  # This is altered for ending hold
            weight=-0.03,  # This is altered for ending hold
            params={  # This is altered for ending hold
                "command_name": "motion",  # This is altered for ending hold
                "sensor_cfg": terminal_contact_cfg,  # This is altered for ending hold
                "body_names": terminal_body_names,  # This is altered for ending hold
                "contact_threshold": 0.0,  # This is altered for ending hold
            },  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_quiet_ankle_joint_velocity = RewTerm(  # This is altered for ending hold
            func=mdp.final_quiet_joint_velocity_l2,  # This is altered for ending hold
            weight=-0.03,  # This is altered for ending hold
            params={"command_name": "motion", "joint_names": T800_ANKLE_JOINT_NAMES},  # This is altered for ending hold
        )  # This is altered for ending hold
        self.rewards.final_contact_stability = RewTerm(  # This is altered for ending hold
            func=mdp.final_contact_stability,  # This is altered for ending hold
            weight=0.5,  # This is altered for ending hold
            params={  # This is altered for ending hold
                "sensor_cfg": terminal_contact_cfg,  # This is altered for ending hold
                "command_name": "motion",  # This is altered for ending hold
                "contact_threshold": 0.0,  # This is altered for ending hold
            },  # This is altered for ending hold
        )  # This is altered for ending hold


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
