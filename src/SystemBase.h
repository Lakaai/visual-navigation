/**
 * @file SystemBase.h
 * @brief Defines the base class for system representation.
 */

#ifndef SYSTEMBASE_H
#define SYSTEMBASE_H

#include <Eigen/Core>

/**
 * @class SystemBase
 * @brief Base class for system representation.
 *
 * This class provides a basic interface for system dynamics and prediction.
 */
class SystemBase
{
public:
    /**
     * @brief Construct a new SystemBase object.
     */
    SystemBase();

    /**
     * @brief Destroy the SystemBase object.
     */
    virtual ~SystemBase();

    /**
     * @brief Predict the system state at a given time.
     * @param time The time to predict the system state for.
     */
    virtual void predict(double time) = 0;

    /**
     * @brief Compute the system dynamics.
     * @param x The current state vector.
     * @return The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x) const = 0;

    /**
     * @brief Compute the system dynamics and its Jacobian.
     * @param x The current state vector.
     * @param J Output parameter for the Jacobian matrix.
     * @return The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const = 0;

protected:
    double time_;  ///< The current system time.
};

#endif
