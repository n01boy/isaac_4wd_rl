#!/usr/bin/env python3
"""
Training script for 4WD robot with PPO algorithm.

This script trains a 4WD robot to navigate using LiDAR sensors with domain randomization
for sim-to-real transfer to Raspberry Pi hardware.

Usage:
    python train.py --num_envs 10 --headless
"""

import argparse
import os
import sys
from datetime import datetime

# Add project to Python path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import torch
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train 4WD robot with PPO")
parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
parser.add_argument("--enable_cameras", type=bool, default=False, help="Enable camera rendering")
parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching app
from envs.isaac_4wd_env import FourWDEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import RSL-RL (PPO implementation)
from rsl_rl.runners import OnPolicyRunner


def main():
    """Main training function."""

    # Set random seed
    torch.manual_seed(args.seed)

    # Create environment configuration
    env_cfg = FourWDEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Enable domain randomization
    if hasattr(env_cfg, 'events'):
        env_cfg.events.reset_robot_position.params = {
            "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},
        }

    # Create environment
    print(f"[INFO] Creating environment with {args.num_envs} parallel instances...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create PPO algorithm configuration
    agent_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,  # Collect 24 steps per environment before update
        max_iterations=args.max_iterations,
        save_interval=50,  # Save checkpoint every 50 iterations

        # PPO algorithm parameters
        algorithm_class_name="PPO",
        empirical_normalization=False,

        # Policy network architecture
        policy=dict(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
        ),

        # PPO-specific parameters
        algorithm=dict(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_path, "logs", f"4wd_ppo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Logs will be saved to: {log_dir}")

    # Create PPO runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    # Load checkpoint if specified
    if args.checkpoint is not None:
        print(f"[INFO] Loading checkpoint from: {args.checkpoint}")
        runner.load(args.checkpoint)

    # Start training
    print("[INFO] Starting training...")
    print(f"[INFO] - Number of environments: {args.num_envs}")
    print(f"[INFO] - Max iterations: {args.max_iterations}")
    print(f"[INFO] - Device: {env.device}")
    print(f"[INFO] - Observation space: {env.observation_space}")
    print(f"[INFO] - Action space: {env.action_space}")
    print("-" * 80)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # Save final model
    final_checkpoint_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_checkpoint_path)
    print(f"[INFO] Training completed! Final model saved to: {final_checkpoint_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation app
        simulation_app.close()
