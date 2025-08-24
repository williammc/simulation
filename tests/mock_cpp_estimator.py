#!/usr/bin/env python3
"""
Mock C++ estimator for testing binary integration.
This script mimics the behavior of a real C++ SLAM estimator.
"""

import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mock C++ SLAM Estimator")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", default="estimation_result.json", help="Output JSON file")
    parser.add_argument("--config", help="Configuration file (ignored)")
    parser.add_argument("--delay", type=float, default=0.1, help="Processing delay in seconds")
    parser.add_argument("--fail", action="store_true", help="Simulate failure")
    parser.add_argument("--timeout", action="store_true", help="Simulate timeout")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise level for estimation")
    return parser.parse_args()


def add_noise(position, noise_level):
    """Add Gaussian noise to a position."""
    if position is None:
        return None
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    if pos.ndim == 0:  # scalar
        return float(pos + np.random.normal(0, noise_level))
    return (pos + np.random.normal(0, noise_level, len(pos))).tolist()


def process_simulation_data(input_file, noise_level=0.01):
    """
    Process simulation data and generate mock estimation results.
    
    Args:
        input_file: Path to input JSON file
        noise_level: Standard deviation of noise to add
    
    Returns:
        Dictionary with estimation results
    """
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract trajectory
    trajectory = data.get("trajectory", [])
    landmarks = data.get("landmarks", [])
    
    # Generate estimated trajectory (ground truth with noise)
    estimated_trajectory = []
    for point in trajectory:
        est_point = {
            "timestamp": point["timestamp"],
            "position": add_noise(point["position"], noise_level)
        }
        
        if "quaternion" in point:
            # Add small noise to quaternion (simplified)
            q = np.array(point["quaternion"])
            q += np.random.normal(0, noise_level * 0.1, 4)
            q = q / np.linalg.norm(q)  # Renormalize
            est_point["quaternion"] = q.tolist()
        
        if "velocity" in point:
            est_point["velocity"] = add_noise(point["velocity"], noise_level * 0.5)
        
        estimated_trajectory.append(est_point)
    
    # Generate estimated landmarks (ground truth with noise)
    estimated_landmarks = []
    for landmark in landmarks:
        est_landmark = {
            "id": landmark["id"],
            "position": add_noise(landmark["position"], noise_level * 2)
        }
        estimated_landmarks.append(est_landmark)
    
    # Generate mock covariances
    num_states = len(estimated_trajectory)
    covariances = {
        "trajectory_covariance": np.eye(num_states * 6) * (noise_level ** 2),
        "landmark_covariance": np.eye(len(landmarks) * 3) * (noise_level * 2) ** 2
    }
    
    # Build output
    output = {
        "metadata": {
            "estimator": "mock_cpp",
            "version": "1.0",
            "runtime_ms": int(time.time() * 1000) % 10000,
            "noise_level": noise_level
        },
        "estimated_trajectory": estimated_trajectory,
        "estimated_landmarks": estimated_landmarks,
        "covariances": {
            "trajectory_diagonal": np.diag(covariances["trajectory_covariance"]).tolist()[:100],
            "landmark_diagonal": np.diag(covariances["landmark_covariance"]).tolist()[:100]
        }
    }
    
    return output


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Simulate timeout if requested
    if args.timeout:
        time.sleep(1000)  # Sleep forever (will be killed by timeout)
        return 1
    
    # Simulate failure if requested
    if args.fail:
        print("Error: Simulated failure", file=sys.stderr)
        return 1
    
    # Simulate processing delay
    time.sleep(args.delay)
    
    try:
        # Process the data
        output = process_simulation_data(args.input, args.noise)
        
        # Write output
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Successfully processed {args.input}")
        print(f"Output written to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())