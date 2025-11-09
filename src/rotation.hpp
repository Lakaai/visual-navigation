#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    // TODO: Lab 7
    Scalar c = cos(x);
    Scalar s = sin(x);
    R(1,1) = c; R(1,2) = -s;
    R(2,1) = s; R(2,2) = c;

    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx            =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(1,1)       = -sin(x);
    dRdx(2,1)       =  cos(x);

    dRdx(1,2)       = -cos(x);
    dRdx(2,2)       = -sin(x);
    return rotx(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    // TODO: Lab 7
    Scalar c = cos(x);
    Scalar s = sin(x);
    R(0,0) = c;     R(0,2) = s;
    R(2,0) = -s;    R(2,2) = c;

    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(2,0)    = -cos(x);

    dRdx(0,2)    =  cos(x);
    dRdx(2,2)    = -sin(x);
    return roty(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    // TODO: Lab 7
    Scalar c = cos(x);
    Scalar s = sin(x);

    R(0,0) =  c;  R(0,1) = -s;
    R(1,0) =  s;  R(1,1) =  c;
    
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(1,0)    =  cos(x);

    dRdx(0,1)    = -cos(x);
    dRdx(1,1)    = -sin(x);
    return rotz(x);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> rpy2rot(const Eigen::MatrixBase<Derived> & Theta)
{
    using Scalar = typename Derived::Scalar;
    // R = Rz*Ry*Rx
    Eigen::Matrix3<Scalar> R;
    // TODO: Lab 7
    Scalar roll = Theta[0];
    Scalar pitch = Theta[1];
    Scalar yaw = Theta[2];

    Eigen::Matrix3<Scalar> Rx = rotx(roll);
    Eigen::Matrix3<Scalar> Ry = roty(pitch);
    Eigen::Matrix3<Scalar> Rz = rotz(yaw);

    R = Rz * Ry * Rx;

    return R;
}

template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> rot2rpy(const Eigen::MatrixBase<Derived> & R)
{
    using Scalar = typename Derived::Scalar;
    using std::atan2, std::hypot;
    Eigen::Vector3<Scalar> Theta;
        // TODO: Lab 7
    Scalar pitch = atan2(-R(2,0), hypot(R(0,0), R(1,0)));
    
    if (hypot(R(0,0), R(1,0)) > Scalar(1e-10)) {
        // Not in gimbal lock
        Theta[0] = atan2(R(2,1), R(2,2));  // Roll
        Theta[1] = pitch;                  // Pitch
        Theta[2] = atan2(R(1,0), R(0,0));  // Yaw
    } else {
        // Gimbal lock case
        Theta[0] = Scalar(0);              // Roll (arbitrary)
        Theta[1] = pitch;                  // Pitch
        Theta[2] = atan2(-R(0,1), R(1,1)); // Yaw
    }
    return Theta;
}

#endif
