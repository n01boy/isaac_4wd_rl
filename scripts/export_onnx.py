#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format for Raspberry Pi deployment.

This script converts the trained PPO policy network to ONNX format, which can be
efficiently executed on Raspberry Pi using ONNX Runtime.

Usage:
    python export_onnx.py --checkpoint logs/model_final.pt --output models/4wd_policy.onnx
"""

import argparse
import os
import sys

# Add project to Python path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np


def load_policy_from_checkpoint(checkpoint_path):
    """Load the policy network from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract policy network state dict
    if 'model_state_dict' in checkpoint:
        policy_state_dict = checkpoint['model_state_dict']
    elif 'policy' in checkpoint:
        policy_state_dict = checkpoint['policy']
    else:
        raise ValueError("Cannot find policy in checkpoint. Keys available: " + str(checkpoint.keys()))

    return policy_state_dict


class SimplifiedPolicy(nn.Module):
    """Simplified policy network for ONNX export (actor only, no critic)."""

    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 128], activation='elu'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Activation function
        if activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build actor network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Action outputs in [-1, 1]

        self.actor = nn.Sequential(*layers)

    def forward(self, observations):
        """Forward pass - takes observations and returns actions."""
        return self.actor(observations)


def export_to_onnx(checkpoint_path, output_path, observation_dim=363, action_dim=4):
    """
    Export trained policy to ONNX format.

    Args:
        checkpoint_path: Path to the saved PyTorch checkpoint
        output_path: Output path for ONNX model
        observation_dim: Dimension of observation space (360 LiDAR + 2 velocity + 1 angular velocity)
        action_dim: Dimension of action space (4 wheel velocities)
    """

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create simplified policy model
    print(f"[INFO] Creating policy network (input={observation_dim}, output={action_dim})")
    policy = SimplifiedPolicy(
        input_dim=observation_dim,
        output_dim=action_dim,
        hidden_dims=[256, 256, 128],
        activation='elu'
    )

    # Load weights from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'actor_critic' in checkpoint:
        state_dict = checkpoint['actor_critic']
    else:
        # Try to load directly
        state_dict = checkpoint

    # Filter only actor weights
    actor_state_dict = {}
    for key, value in state_dict.items():
        if 'actor' in key:
            new_key = key.replace('actor.', '').replace('policy.', '')
            actor_state_dict[new_key] = value

    policy.actor.load_state_dict(actor_state_dict, strict=False)
    policy.eval()

    print("[INFO] Model loaded successfully")

    # Create dummy input
    dummy_input = torch.randn(1, observation_dim)

    # Test forward pass
    with torch.no_grad():
        dummy_output = policy(dummy_input)
        print(f"[INFO] Test forward pass successful. Output shape: {dummy_output.shape}")

    # Export to ONNX
    print(f"[INFO] Exporting to ONNX format: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    print("[INFO] ONNX export completed")

    # Verify ONNX model
    print("[INFO] Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model is valid")

    # Test ONNX Runtime inference
    print("[INFO] Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare outputs
    torch_output = dummy_output.numpy()
    onnx_output = ort_outputs[0]
    max_diff = np.abs(torch_output - onnx_output).max()

    print(f"[INFO] Max difference between PyTorch and ONNX: {max_diff:.6f}")

    if max_diff < 1e-5:
        print("[SUCCESS] ONNX model matches PyTorch model!")
    else:
        print("[WARNING] Outputs differ - please verify the conversion")

    # Print model info
    print("\n" + "="*80)
    print("ONNX Model Information:")
    print("="*80)
    print(f"Input name: {ort_session.get_inputs()[0].name}")
    print(f"Input shape: {ort_session.get_inputs()[0].shape}")
    print(f"Output name: {ort_session.get_outputs()[0].name}")
    print(f"Output shape: {ort_session.get_outputs()[0].shape}")
    print(f"Model size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print("="*80)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--obs_dim", type=int, default=363, help="Observation dimension")
    parser.add_argument("--action_dim", type=int, default=4, help="Action dimension")

    args = parser.parse_args()

    try:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            observation_dim=args.obs_dim,
            action_dim=args.action_dim
        )
        print(f"\n[SUCCESS] Model exported successfully to: {args.output}")
        print("\nNext steps:")
        print("1. Transfer the ONNX model to your Raspberry Pi")
        print("2. Install onnxruntime on Raspberry Pi: pip install onnxruntime")
        print("3. Use the inference script in raspberry_pi/inference.py")

    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
