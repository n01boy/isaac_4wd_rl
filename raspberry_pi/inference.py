#!/usr/bin/env python3
"""
Raspberry Pi Inference Script for 4WD Robot

This script runs the trained ONNX model on Raspberry Pi hardware with real LiDAR sensor.

Hardware Requirements:
- Raspberry Pi 4/5
- RPLIDAR A1/A2 or compatible 2D LiDAR
- L298N or similar motor driver
- 4WD robot chassis

Dependencies:
    pip install onnxruntime numpy rplidar-roboticia

Usage:
    sudo python3 inference.py --model 4wd_policy.onnx
"""

import argparse
import time
import signal
import sys
from collections import deque

import numpy as np
import onnxruntime as ort

try:
    from rplidar import RPLidar
except ImportError:
    print("[WARNING] rplidar library not installed. Install with: pip install rplidar-roboticia")
    RPLidar = None

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("[WARNING] RPi.GPIO not available. Using simulation mode.")
    GPIO = None


class MotorController:
    """Controls 4 DC motors using L298N or similar H-bridge driver."""

    def __init__(self, enable_pins, input_pins):
        """
        Initialize motor controller.

        Args:
            enable_pins: List of 4 PWM enable pins [FL, FR, RL, RR]
            input_pins: List of 8 direction pins [[FL1, FL2], [FR1, FR2], [RL1, RL2], [RR1, RR2]]
        """
        self.enable_pins = enable_pins
        self.input_pins = input_pins
        self.pwm_objects = []

        if GPIO is not None:
            GPIO.setmode(GPIO.BCM)

            # Setup enable pins (PWM)
            for pin in enable_pins:
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, 1000)  # 1kHz PWM frequency
                pwm.start(0)
                self.pwm_objects.append(pwm)

            # Setup direction pins
            for pin_pair in input_pins:
                for pin in pin_pair:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)

            print("[INFO] Motor controller initialized")

    def set_motor_speed(self, motor_idx, speed):
        """
        Set speed for a single motor.

        Args:
            motor_idx: Motor index (0=FL, 1=FR, 2=RL, 3=RR)
            speed: Speed value in range [-1.0, 1.0]
        """
        if GPIO is None:
            return

        # Clamp speed
        speed = np.clip(speed, -1.0, 1.0)

        # Set direction
        in1, in2 = self.input_pins[motor_idx]
        if speed > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        # Set PWM duty cycle (0-100%)
        duty_cycle = abs(speed) * 100
        self.pwm_objects[motor_idx].ChangeDutyCycle(duty_cycle)

    def set_all_motors(self, speeds):
        """Set speeds for all 4 motors at once."""
        for idx, speed in enumerate(speeds):
            self.set_motor_speed(idx, speed)

    def stop_all(self):
        """Stop all motors."""
        self.set_all_motors([0, 0, 0, 0])

    def cleanup(self):
        """Cleanup GPIO resources."""
        self.stop_all()
        if GPIO is not None:
            for pwm in self.pwm_objects:
                pwm.stop()
            GPIO.cleanup()


class LiDARProcessor:
    """Processes LiDAR data into format expected by the neural network."""

    def __init__(self, num_points=360, max_distance=12.0, min_distance=0.15):
        self.num_points = num_points
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.angle_resolution = 360.0 / num_points

    def process_scan(self, scan_data):
        """
        Convert raw LiDAR scan to fixed-size distance array.

        Args:
            scan_data: Iterator of (quality, angle, distance) tuples from RPLidar

        Returns:
            numpy array of shape (360,) with normalized distances
        """
        distances = np.ones(self.num_points) * self.max_distance  # Default to max distance

        for quality, angle, distance in scan_data:
            # Convert distance from mm to meters
            distance_m = distance / 1000.0

            # Bin into appropriate angle index
            angle_idx = int(angle / self.angle_resolution) % self.num_points

            # Clamp to valid range
            distance_m = np.clip(distance_m, self.min_distance, self.max_distance)

            distances[angle_idx] = distance_m

        return distances


class RobotController:
    """Main controller integrating LiDAR, model inference, and motor control."""

    def __init__(self, model_path, lidar_port='/dev/ttyUSB0', control_freq=20):
        """
        Initialize robot controller.

        Args:
            model_path: Path to ONNX model file
            lidar_port: Serial port for LiDAR sensor
            control_freq: Control loop frequency in Hz
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # Load ONNX model
        print(f"[INFO] Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Initialize LiDAR
        if RPLidar is not None:
            print(f"[INFO] Connecting to LiDAR on {lidar_port}")
            self.lidar = RPLidar(lidar_port)
            self.lidar_processor = LiDARProcessor()
        else:
            self.lidar = None
            print("[WARNING] LiDAR not available - using simulated data")

        # Initialize motor controller
        # Pin configuration for L298N (adjust for your hardware)
        enable_pins = [17, 22, 23, 24]  # ENA, ENB, ENC, END
        input_pins = [
            [27, 18],  # Front Left: IN1, IN2
            [15, 14],  # Front Right: IN3, IN4
            [25, 8],   # Rear Left: IN5, IN6
            [7, 1],    # Rear Right: IN7, IN8
        ]
        self.motors = MotorController(enable_pins, input_pins)

        # State tracking
        self.last_velocities = deque(maxlen=5)
        self.running = True

        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n[INFO] Shutting down...")
        self.running = False

    def get_observation(self):
        """Get current observation from sensors."""
        # Get LiDAR scan
        if self.lidar is not None:
            try:
                scan = next(self.lidar.iter_scans())
                lidar_data = self.lidar_processor.process_scan(scan)
            except Exception as e:
                print(f"[WARNING] LiDAR read error: {e}")
                lidar_data = np.ones(360) * 12.0
        else:
            # Simulated LiDAR data
            lidar_data = np.random.uniform(0.5, 12.0, 360)

        # Estimate velocity from previous actions (simplified)
        if len(self.last_velocities) > 1:
            avg_left = np.mean([v[0] + v[2] for v in self.last_velocities]) / 2.0
            avg_right = np.mean([v[1] + v[3] for v in self.last_velocities]) / 2.0
            linear_vel = (avg_left + avg_right) / 2.0 * 0.033  # Convert to m/s (wheel radius)
            angular_vel = (avg_right - avg_left) / 0.13  # Convert to rad/s (track width)
        else:
            linear_vel = 0.0
            angular_vel = 0.0

        # Concatenate observation: [lidar(360), vel_x, vel_y, ang_vel]
        obs = np.concatenate([
            lidar_data,
            [linear_vel, 0.0],  # vel_y is always 0 for wheeled robot
            [angular_vel]
        ]).astype(np.float32)

        return obs

    def predict_action(self, observation):
        """Run inference on the ONNX model."""
        obs_batch = observation.reshape(1, -1)  # Add batch dimension
        ort_inputs = {self.input_name: obs_batch}
        action = self.session.run([self.output_name], ort_inputs)[0]
        return action[0]  # Remove batch dimension

    def run(self):
        """Main control loop."""
        print("[INFO] Starting control loop at {:.1f} Hz".format(self.control_freq))
        print("[INFO] Press Ctrl+C to stop")

        # Start LiDAR motor
        if self.lidar is not None:
            self.lidar.start_motor()
            time.sleep(2)  # Wait for motor to stabilize

        try:
            while self.running:
                loop_start = time.time()

                # Get observation
                obs = self.get_observation()

                # Predict action
                action = self.predict_action(obs)

                # Apply action to motors
                self.motors.set_all_motors(action)
                self.last_velocities.append(action)

                # Print status
                min_distance = np.min(obs[:360])
                print(f"\r[INFO] Min distance: {min_distance:.2f}m | "
                      f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]",
                      end='', flush=True)

                # Maintain control frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)

        finally:
            # Cleanup
            self.motors.stop_all()
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            self.motors.cleanup()
            print("\n[INFO] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Run 4WD robot with trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--lidar_port", type=str, default="/dev/ttyUSB0", help="LiDAR serial port")
    parser.add_argument("--freq", type=int, default=20, help="Control frequency (Hz)")

    args = parser.parse_args()

    # Create and run controller
    controller = RobotController(
        model_path=args.model,
        lidar_port=args.lidar_port,
        control_freq=args.freq
    )

    controller.run()


if __name__ == "__main__":
    main()
