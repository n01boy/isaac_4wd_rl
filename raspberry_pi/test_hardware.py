#!/usr/bin/env python3
"""
Hardware Test Script for Raspberry Pi 4WD Robot

This script tests individual hardware components before running the full inference.

Usage:
    sudo python3 test_hardware.py --test [motors|lidar|all]
"""

import argparse
import time
import sys

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("[ERROR] RPi.GPIO not installed. Install with: sudo pip install RPi.GPIO")
    GPIO = None

try:
    from rplidar import RPLidar
except ImportError:
    print("[ERROR] rplidar not installed. Install with: pip install rplidar-roboticia")
    RPLidar = None


def test_motors():
    """Test motor controller by running each motor individually."""
    print("\n" + "="*60)
    print("MOTOR TEST")
    print("="*60)

    if GPIO is None:
        print("[ERROR] Cannot test motors without RPi.GPIO")
        return False

    # Pin configuration (adjust for your hardware)
    enable_pins = [17, 22, 23, 24]
    input_pins = [
        [27, 18],  # Front Left
        [15, 14],  # Front Right
        [25, 8],   # Rear Left
        [7, 1],    # Rear Right
    ]

    motor_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup pins
        pwm_objects = []
        for pin in enable_pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, 1000)
            pwm.start(0)
            pwm_objects.append(pwm)

        for pin_pair in input_pins:
            for pin in pin_pair:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

        print("[INFO] GPIO initialized successfully")

        # Test each motor
        for idx in range(4):
            print(f"\n[TEST] Testing {motor_names[idx]}...")
            print("       Motor should spin FORWARD for 2 seconds")

            # Forward
            GPIO.output(input_pins[idx][0], GPIO.HIGH)
            GPIO.output(input_pins[idx][1], GPIO.LOW)
            pwm_objects[idx].ChangeDutyCycle(50)  # 50% speed
            time.sleep(2)

            # Stop
            pwm_objects[idx].ChangeDutyCycle(0)
            GPIO.output(input_pins[idx][0], GPIO.LOW)
            GPIO.output(input_pins[idx][1], GPIO.LOW)
            time.sleep(1)

            print(f"       Motor should spin BACKWARD for 2 seconds")

            # Backward
            GPIO.output(input_pins[idx][0], GPIO.LOW)
            GPIO.output(input_pins[idx][1], GPIO.HIGH)
            pwm_objects[idx].ChangeDutyCycle(50)
            time.sleep(2)

            # Stop
            pwm_objects[idx].ChangeDutyCycle(0)
            GPIO.output(input_pins[idx][0], GPIO.LOW)
            GPIO.output(input_pins[idx][1], GPIO.LOW)
            time.sleep(1)

            user_input = input(f"       Did {motor_names[idx]} work correctly? (y/n): ")
            if user_input.lower() != 'y':
                print(f"[FAIL] {motor_names[idx]} test failed")
                return False

        print("\n[PASS] All motors working correctly!")
        return True

    except Exception as e:
        print(f"[ERROR] Motor test failed: {e}")
        return False

    finally:
        # Cleanup
        for pwm in pwm_objects:
            pwm.stop()
        GPIO.cleanup()


def test_lidar(port='/dev/ttyUSB0', duration=10):
    """Test LiDAR sensor by reading scans."""
    print("\n" + "="*60)
    print("LIDAR TEST")
    print("="*60)

    if RPLidar is None:
        print("[ERROR] Cannot test LiDAR without rplidar library")
        return False

    try:
        print(f"[INFO] Connecting to LiDAR on {port}...")
        lidar = RPLidar(port)

        # Get device info
        info = lidar.get_info()
        print(f"[INFO] LiDAR Info:")
        print(f"       Model: {info['model']}")
        print(f"       Firmware: {info['firmware']}")
        print(f"       Hardware: {info['hardware']}")

        # Get health status
        health = lidar.get_health()
        print(f"[INFO] Health Status: {health}")

        # Start motor
        print(f"[INFO] Starting LiDAR motor...")
        lidar.start_motor()
        time.sleep(2)

        # Read scans
        print(f"[INFO] Reading scans for {duration} seconds...")
        print(f"       Press Ctrl+C to stop early")

        start_time = time.time()
        scan_count = 0

        for scan in lidar.iter_scans(max_buf_meas=500):
            scan_count += 1
            num_points = len(scan)

            # Find min/max distances
            distances = [point[2] for point in scan]
            min_dist = min(distances) if distances else 0
            max_dist = max(distances) if distances else 0

            print(f"\r[SCAN {scan_count}] Points: {num_points:3d} | "
                  f"Min: {min_dist:6.1f}mm | Max: {max_dist:6.1f}mm",
                  end='', flush=True)

            if time.time() - start_time > duration:
                break

        print(f"\n[PASS] LiDAR test completed successfully!")
        print(f"       Total scans: {scan_count}")
        print(f"       Average rate: {scan_count/duration:.1f} Hz")

        return True

    except Exception as e:
        print(f"\n[ERROR] LiDAR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except:
            pass


def test_all():
    """Run all hardware tests."""
    print("\n" + "="*60)
    print("FULL HARDWARE TEST SUITE")
    print("="*60)

    results = {}

    # Test LiDAR first (non-intrusive)
    results['lidar'] = test_lidar()

    # Test motors
    print("\n[INFO] Now testing motors...")
    print("[WARNING] Make sure the robot is elevated or has free space to move!")
    input("Press ENTER to continue with motor test, or Ctrl+C to skip...")
    results['motors'] = test_motors()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{component.upper():15s}: [{status}]")

    all_passed = all(results.values())
    print("="*60)

    if all_passed:
        print("[SUCCESS] All hardware tests passed!")
        print("[INFO] You can now run: sudo python3 inference.py --model <model.onnx>")
    else:
        print("[FAILURE] Some tests failed. Please check hardware connections.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test 4WD robot hardware")
    parser.add_argument("--test", type=str, choices=['motors', 'lidar', 'all'], default='all',
                        help="Which component to test")
    parser.add_argument("--lidar_port", type=str, default="/dev/ttyUSB0",
                        help="LiDAR serial port")
    parser.add_argument("--duration", type=int, default=10,
                        help="LiDAR test duration in seconds")

    args = parser.parse_args()

    try:
        if args.test == 'motors':
            success = test_motors()
        elif args.test == 'lidar':
            success = test_lidar(args.lidar_port, args.duration)
        else:
            success = test_all()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        if GPIO is not None:
            GPIO.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
