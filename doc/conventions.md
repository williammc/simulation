# Coding Conventions and Standards

## Transformation Naming Convention

This project strictly follows a consistent naming convention for all coordinate transformations to ensure clarity and prevent errors.

### Core Principle

The notation `A_X_B` means: **"X that transforms FROM frame B TO frame A"**

### Standard Notation

#### Rotation Matrices
```python
A_R_B  # 3x3 rotation matrix from frame B to frame A
# Usage: p_A = A_R_B @ p_B
```

#### Translation Vectors
```python
A_t_B  # 3x1 translation vector from frame B to frame A
# Usage: p_A = A_R_B @ p_B + A_t_B
```

#### Transformation Matrices (SE3)
```python
A_T_B  # 4x4 homogeneous transformation from frame B to frame A
# Structure:
# A_T_B = | A_R_B  A_t_B |
#         |   0      1   |
# Usage: p_A_hom = A_T_B @ p_B_hom
```

#### Quaternions
```python
A_q_B  # Quaternion [w, x, y, z] representing rotation from B to A
# Convention: w is the scalar part (first element)
```

### Transformation Chaining

Transformations chain by matrix multiplication:
```python
# To go from C to A through B:
A_T_C = A_T_B @ B_T_C

# This reads naturally: A-from-C = A-from-B × B-from-C
```

### Common Frame Definitions

| Symbol | Frame | Description | Origin |
|--------|-------|-------------|--------|
| **W** | World | Fixed inertial reference frame | Arbitrary fixed point |
| **M** | Map | SLAM map reference frame | First pose or loop closure |
| **B** | Body | Vehicle/robot body frame | IMU center (typically) |
| **I** | IMU | Inertial measurement unit | IMU sensor center¹ |
| **C** | Camera | Generic camera frame | Optical center |
| **C0** | Camera 0 | Primary/left camera | Left optical center |
| **C1** | Camera 1 | Secondary/right camera | Right optical center |
| **E** | Event | Event camera frame | Event camera center |
| **L** | Lidar | Lidar sensor frame | Lidar center |
| **G** | GPS | GPS antenna frame | Antenna phase center |

¹ **IMU Origin Convention**: The IMU sensor center represents a unified origin for both accelerometer and gyroscope. This reflects modern MEMS IMUs where both sensors are integrated on the same chip with negligible separation (<1mm). No lever arm compensation is applied between accelerometer and gyroscope measurements.

### Practical Examples

#### Example 1: Camera to World Transformation
```python
# Given: Camera observes a landmark
p_C = np.array([0, 0, 5])  # 5 meters in front of camera

# Known calibration and pose
B_T_C = load_calibration()  # Body-from-camera (extrinsic calibration)
W_T_B = get_current_pose()  # World-from-body (vehicle pose)

# Transform to world frame
W_T_C = W_T_B @ B_T_C
p_W = transform_point(W_T_C, p_C)
```

#### Example 2: IMU to Camera Time Alignment
```python
# IMU and camera have different timestamps
t_imu = 1000.0  # IMU timestamp
t_cam = 1000.1  # Camera timestamp

# Interpolate IMU pose at camera time
W_T_B_imu = get_pose_at_time(t_imu)
W_T_B_cam = interpolate_pose(W_T_B_imu, t_cam)

# Now can project with synchronized pose
W_T_C = W_T_B_cam @ B_T_C
```

#### Example 3: Stereo Camera Setup
```python
# Stereo baseline
C0_T_C1 = get_stereo_calibration()  # Left-from-right transform

# Point in right camera
p_C1 = np.array([u1, v1, depth])

# Transform to left camera
p_C0 = transform_point(C0_T_C1, p_C1)

# Transform to world
W_T_C0 = W_T_B @ B_T_C0
p_W = transform_point(W_T_C0, p_C0)
```

### Inverse Transformations

The inverse transformation reverses the direction:
```python
B_T_A = se3_inverse(A_T_B)  # B-from-A is inverse of A-from-B
B_R_A = A_R_B.T             # For rotation matrices
B_q_A = quaternion_inverse(A_q_B)  # For quaternions
```

### Variable Naming Guidelines

#### Good Examples ✓
```python
W_T_B_init      # Initial world-from-body transform
C_R_B_calib     # Calibrated camera-from-body rotation
W_t_L_landmark  # World-from-landmark translation
B_T_C0_left     # Body-from-left-camera transform
```

#### Bad Examples ✗
```python
T_wb        # Ambiguous direction
R_cam       # Missing frame information  
transform   # No frame specification
T_c2w       # Inconsistent notation
```

### Jacobian Naming

For derivatives and Jacobians:
```python
J_A_X_B  # Jacobian of X with respect to B, expressed in frame A
# Example:
J_W_p_theta  # Jacobian of position w.r.t. angle, in world frame
```

### Covariance Naming

For uncertainty representation:
```python
Sigma_A_XX  # Covariance of X expressed in frame A
# Example:
Sigma_W_pp  # Position covariance in world frame
Sigma_B_vv  # Velocity covariance in body frame
```

## IMU-Specific Conventions

### IMU Sensor Model
The IMU is modeled as a single integrated MEMS sensor unit:

- **Unified Origin**: Both accelerometer and gyroscope measurements originate from the same physical point (IMU center)
- **Measurement Frame**: All IMU measurements are expressed in the sensor frame (S), which typically aligns with the body frame (B)
- **Extrinsics**: The transformation `B_T_S` (body-from-sensor) defaults to identity, meaning IMU and body frames are aligned
- **No Lever Arm**: No compensation for spatial separation between accelerometer and gyroscope (assumed co-located)

This model is appropriate for:
- Modern MEMS IMUs (BMI055, ICM-20649, MPU-9250)
- Tactical-grade IMUs with integrated sensors
- Most robotics and drone applications

For high-precision applications with physically separated accelerometer and gyroscope units, additional lever arm modeling would be required.

### IMU Measurement Convention
```python
# IMU measurements in sensor frame
accelerometer: np.ndarray  # [ax, ay, az] in m/s²
gyroscope: np.ndarray      # [wx, wy, wz] in rad/s

# Transform to body frame (typically identity)
B_T_S = imu_calibration.extrinsics.B_T_S  # Usually I (identity)
accel_body = B_T_S[:3, :3] @ accelerometer
gyro_body = B_T_S[:3, :3] @ gyroscope
```

## Additional Conventions

### Units
- **Distance**: meters [m]
- **Angles**: radians [rad] (degrees only in config/display)
- **Time**: seconds [s]
- **Frequency**: Hertz [Hz]
- **Angular velocity**: radians/second [rad/s]
- **Linear velocity**: meters/second [m/s]
- **Acceleration**: meters/second² [m/s²]

### Matrix Storage
- **Row-major** order (NumPy default)
- **Column vectors** for positions/directions
- **Homogeneous coordinates**: [x, y, z, 1]ᵀ

### Quaternion Convention
- **Order**: [w, x, y, z] where w is scalar
- **Normalization**: Always unit quaternions
- **Hamilton convention** for multiplication

### Time Conventions
- **Timestamps**: Unix epoch (seconds since 1970-01-01)
- **Monotonic**: Strictly increasing
- **Synchronization**: Sub-millisecond accuracy required

## Validation Functions

```python
def validate_transformation_name(name: str) -> bool:
    """
    Check if a variable name follows our convention.
    Valid format: A_X_B where A,B are frames and X is R/t/T/q
    """
    import re
    pattern = r'^[A-Z][0-9]?_[RtTq]_[A-Z][0-9]?(_\w+)?$'
    return bool(re.match(pattern, name))

# Examples
assert validate_transformation_name("W_T_B")        # ✓
assert validate_transformation_name("C0_R_C1")      # ✓ 
assert validate_transformation_name("W_T_B_init")   # ✓
assert not validate_transformation_name("T_wb")     # ✗
assert not validate_transformation_name("transform") # ✗
```

## References

This convention is inspired by:
- [Robotics: Modelling, Planning and Control](https://link.springer.com/book/10.1007/978-1-84628-642-1) by Siciliano et al.
- [A Micro Lie Theory for State Estimation in Robotics](https://arxiv.org/abs/1812.01537) by Solà et al.
- ROS [REP-103](https://www.ros.org/reps/rep-0103.html) and [REP-105](https://www.ros.org/reps/rep-0105.html)