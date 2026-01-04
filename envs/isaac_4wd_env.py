"""
4WD Robot Environment for Isaac Lab with LiDAR-based Obstacle Avoidance

This environment simulates a 4-wheel drive robot navigating with LiDAR sensors
for eventual deployment on Raspberry Pi hardware.
"""

import math
import torch
import numpy as np
from typing import Dict, Tuple

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms


##
# Scene Configuration
##

@configclass
class FourWDSceneCfg(InteractiveSceneCfg):
    """Configuration for the 4WD robot scene."""

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # 4WD Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/workspace/isaac_4wd_rl/assets/robots/4wd_vehicle.urdf",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            joint_pos={
                "front_left_wheel_joint": 0.0,
                "front_right_wheel_joint": 0.0,
                "rear_left_wheel_joint": 0.0,
                "rear_right_wheel_joint": 0.0,
            },
        ),
        actuators={
            "wheels": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                velocity_limit=10.0,
                effort_limit=0.5,
                stiffness=0.0,
                damping=0.1,
            ),
        },
    )

    # LiDAR Sensor (360-degree raycast)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/lidar_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=1.0,  # 1 degree resolution = 360 points
        ),
        max_distance=12.0,
        drift_range=(0.0, 0.0),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # Lights
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))


##
# MDP Settings
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy learning."""

        # LiDAR scan data (360 points)
        lidar_data = ObsTerm(func=get_lidar_data)

        # Robot velocity
        base_lin_vel = ObsTerm(func=get_base_velocity)
        base_ang_vel = ObsTerm(func=get_base_angular_velocity)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 4 wheel velocities (continuous action space)
    wheel_velocities = sim_utils.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["front_left_wheel_joint", "front_right_wheel_joint",
                     "rear_left_wheel_joint", "rear_right_wheel_joint"],
        scale=10.0,  # Max velocity in rad/s
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Positive reward for forward progress
    forward_progress = RewTerm(func=reward_forward_progress, weight=1.0)

    # Penalty for collision
    collision = RewTerm(func=reward_collision_penalty, weight=-100.0)

    # Penalty for sharp steering (encourage smooth motion)
    smooth_steering = RewTerm(func=reward_smooth_steering, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=time_out, time_out=True)

    # Collision with obstacles
    collision = DoneTerm(func=collision_termination)


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Reset robot position
    reset_robot_position = EventTerm(
        func=reset_robot_to_default,
        mode="reset",
    )


##
# Environment Configuration
##

@configclass
class FourWDEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the 4WD robot learning environment."""

    # Scene settings
    scene: FourWDSceneCfg = FourWDSceneCfg(num_envs=10, env_spacing=5.0)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Environment settings
    episode_length_s = 25.0  # 25 seconds per episode
    decimation = 2  # Control frequency: 50Hz / 2 = 25Hz
    sim_dt = 1.0 / 50.0  # Simulation timestep: 50Hz

    def __post_init__(self):
        """Post initialization."""
        self.sim.dt = self.sim_dt
        self.sim.render_interval = self.decimation


##
# Observation Functions
##

def get_lidar_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("lidar")) -> torch.Tensor:
    """Get LiDAR distance measurements with optional noise for domain randomization."""
    lidar: RayCaster = env.scene[asset_cfg.name]
    distances = lidar.data.ray_distances[:, :, 0]  # Shape: (num_envs, 360)

    # Add Gaussian noise for sim-to-real transfer (2-5% std dev)
    if env.cfg.scene.lidar.drift_range != (0.0, 0.0):
        noise_std = 0.035  # 3.5% noise
        noise = torch.randn_like(distances) * noise_std * distances
        distances = distances + noise

    # Clip to valid range [0.15, 12.0]
    distances = torch.clamp(distances, 0.15, 12.0)

    return distances


def get_base_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get the linear velocity of the robot base."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b[:, :2]  # Only x, y components


def get_base_angular_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get the angular velocity (yaw rate) of the robot base."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b[:, 2:3]  # Only z component (yaw)


##
# Reward Functions
##

def reward_forward_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for forward velocity."""
    robot: Articulation = env.scene[asset_cfg.name]
    forward_vel = robot.data.root_lin_vel_b[:, 0]  # X-axis velocity
    return forward_vel * env.step_dt


def reward_collision_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty when robot collides (detected via LiDAR minimum distance)."""
    lidar: RayCaster = env.scene["lidar"]
    min_distance = torch.min(lidar.data.ray_distances[:, :, 0], dim=1)[0]
    collision = min_distance < 0.2  # Collision threshold: 20cm
    return collision.float()


def reward_smooth_steering(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for large differences in wheel velocities (encourages smooth turns)."""
    actions = env.action_manager.action
    left_wheels = (actions[:, 0] + actions[:, 2]) / 2.0  # Average of left wheels
    right_wheels = (actions[:, 1] + actions[:, 3]) / 2.0  # Average of right wheels
    steering_diff = torch.abs(left_wheels - right_wheels)
    return steering_diff


##
# Termination Functions
##

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode timeout."""
    return env.episode_length_buf >= env.max_episode_length


def collision_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate episode on collision."""
    lidar: RayCaster = env.scene["lidar"]
    min_distance = torch.min(lidar.data.ray_distances[:, :, 0], dim=1)[0]
    return min_distance < 0.15  # Hard collision threshold


##
# Event Functions
##

def reset_robot_to_default(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset robot to random positions in the arena."""
    robot: Articulation = env.scene[asset_cfg.name]

    # Random positions within 5m x 5m area
    num_resets = len(env_ids)
    positions = torch.zeros((num_resets, 3), device=env.device)
    positions[:, 0] = torch.rand(num_resets, device=env.device) * 4.0 - 2.0  # x: [-2, 2]
    positions[:, 1] = torch.rand(num_resets, device=env.device) * 4.0 - 2.0  # y: [-2, 2]
    positions[:, 2] = 0.1  # z: 10cm above ground

    # Random yaw orientation
    orientations = torch.zeros((num_resets, 4), device=env.device)
    yaw = torch.rand(num_resets, device=env.device) * 2.0 * math.pi
    orientations[:, 0] = torch.cos(yaw / 2.0)  # w
    orientations[:, 3] = torch.sin(yaw / 2.0)  # z

    # Set robot state
    robot.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=1),
        env_ids=env_ids
    )
    robot.reset(env_ids)
