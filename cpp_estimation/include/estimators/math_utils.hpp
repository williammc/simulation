#ifndef ESTIMATORS_MATH_UTILS_HPP
#define ESTIMATORS_MATH_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace estimators {

/**
 * SO(3) exponential map: convert axis-angle to rotation matrix
 * omega: 3x1 axis-angle vector
 * Returns: 3x3 rotation matrix
 */
template<typename FLOAT>
Eigen::Matrix<FLOAT, 3, 3> so3_exp(const Eigen::Matrix<FLOAT, 3, 1>& omega) {
    FLOAT theta = omega.norm();
    
    if (theta < static_cast<FLOAT>(1e-8)) {
        // First-order approximation for small angles
        Eigen::Matrix<FLOAT, 3, 3> R = Eigen::Matrix<FLOAT, 3, 3>::Identity();
        R(0, 1) = -omega(2);
        R(0, 2) = omega(1);
        R(1, 0) = omega(2);
        R(1, 2) = -omega(0);
        R(2, 0) = -omega(1);
        R(2, 1) = omega(0);
        return R;
    }
    
    // Rodrigues' formula
    Eigen::Matrix<FLOAT, 3, 1> axis = omega / theta;
    Eigen::Matrix<FLOAT, 3, 3> K;
    K << 0, -axis(2), axis(1),
         axis(2), 0, -axis(0),
         -axis(1), axis(0), 0;
    
    return Eigen::Matrix<FLOAT, 3, 3>::Identity() + 
           std::sin(theta) * K + 
           (static_cast<FLOAT>(1) - std::cos(theta)) * K * K;
}

/**
 * SO(3) logarithm map: convert rotation matrix to axis-angle
 * R: 3x3 rotation matrix
 * Returns: 3x1 axis-angle vector
 */
template<typename FLOAT>
Eigen::Matrix<FLOAT, 3, 1> so3_log(const Eigen::Matrix<FLOAT, 3, 3>& R) {
    FLOAT trace = R.trace();
    
    // Check for identity matrix
    if (std::abs(trace - static_cast<FLOAT>(3)) < static_cast<FLOAT>(1e-8)) {
        return Eigen::Matrix<FLOAT, 3, 1>::Zero();
    }
    
    // Compute angle
    FLOAT theta = std::acos(std::max(static_cast<FLOAT>(-1), 
                                     std::min(static_cast<FLOAT>(1), 
                                             (trace - static_cast<FLOAT>(1)) / static_cast<FLOAT>(2))));
    
    if (std::abs(theta) < static_cast<FLOAT>(1e-8)) {
        // Small angle approximation
        return Eigen::Matrix<FLOAT, 3, 1>(
            R(2, 1) - R(1, 2),
            R(0, 2) - R(2, 0),
            R(1, 0) - R(0, 1)
        ) / static_cast<FLOAT>(2);
    }
    
    // General case
    FLOAT factor = theta / (static_cast<FLOAT>(2) * std::sin(theta));
    return Eigen::Matrix<FLOAT, 3, 1>(
        R(2, 1) - R(1, 2),
        R(0, 2) - R(2, 0),
        R(1, 0) - R(0, 1)
    ) * factor;
}

/**
 * Skew symmetric matrix from 3D vector
 */
template<typename FLOAT>
Eigen::Matrix<FLOAT, 3, 3> skew_symmetric(const Eigen::Matrix<FLOAT, 3, 1>& v) {
    Eigen::Matrix<FLOAT, 3, 3> S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

/**
 * Convert quaternion to rotation matrix
 * q: [x, y, z, w] quaternion
 */
template<typename FLOAT>
Eigen::Matrix<FLOAT, 3, 3> quaternion_to_rotation_matrix(const Eigen::Matrix<FLOAT, 4, 1>& q) {
    Eigen::Quaternion<FLOAT> quat(q(3), q(0), q(1), q(2)); // w, x, y, z
    return quat.toRotationMatrix();
}

/**
 * Convert rotation matrix to quaternion
 * Returns: [x, y, z, w] quaternion
 */
template<typename FLOAT>
Eigen::Matrix<FLOAT, 4, 1> rotation_matrix_to_quaternion(const Eigen::Matrix<FLOAT, 3, 3>& R) {
    Eigen::Quaternion<FLOAT> q(R);
    return Eigen::Matrix<FLOAT, 4, 1>(q.x(), q.y(), q.z(), q.w());
}

/**
 * Huber robust weight function
 */
template<typename FLOAT>
FLOAT huber_weight(FLOAT residual_norm, FLOAT delta = static_cast<FLOAT>(1.0)) {
    if (residual_norm <= delta) {
        return static_cast<FLOAT>(1.0);
    } else {
        return delta / residual_norm;
    }
}

/**
 * Cauchy robust weight function
 */
template<typename FLOAT>
FLOAT cauchy_weight(FLOAT residual_norm, FLOAT c = static_cast<FLOAT>(2.3849)) {
    return static_cast<FLOAT>(1.0) / (static_cast<FLOAT>(1.0) + (residual_norm / c) * (residual_norm / c));
}

/**
 * Apply SE(3) transformation to a point
 */
template<typename FLOAT>
Eigen::Matrix<FLOAT, 3, 1> transform_point(
    const Eigen::Matrix<FLOAT, 3, 3>& R,
    const Eigen::Matrix<FLOAT, 3, 1>& t,
    const Eigen::Matrix<FLOAT, 3, 1>& point) {
    return R * point + t;
}

/**
 * Inverse SE(3) transformation
 */
template<typename FLOAT>
void inverse_se3(const Eigen::Matrix<FLOAT, 3, 3>& R,
                 const Eigen::Matrix<FLOAT, 3, 1>& t,
                 Eigen::Matrix<FLOAT, 3, 3>& R_inv,
                 Eigen::Matrix<FLOAT, 3, 1>& t_inv) {
    R_inv = R.transpose();
    t_inv = -R_inv * t;
}

} // namespace estimators

#endif // ESTIMATORS_MATH_UTILS_HPP