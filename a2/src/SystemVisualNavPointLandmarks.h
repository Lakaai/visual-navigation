#ifndef SYSTEMVISUALNAVPOINTLANDMARKS_H
#define SYSTEMVISUALNAVPOINTLANDMARKS_H

#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include <map>

/*
 * State containing body velocities, body pose and landmark positions
 *
 *     [ vBNb     ]  Body translational velocity (body-fixed)
 *     [ omegaBNb ]  Body angular velocity (body-fixed)
 *     [ rBNn     ]  Body position (world-fixed)
 * x = [ Thetanb  ]  Body orientation (world-fixed)
 *     [ rL1Nn    ]  Landmark 1 position (world-fixed)
 *     [ rL2Nn    ]  Landmark 2 position (world-fixed)
 *     [ ...      ]  ...
 *
 */
class SystemVisualNavPointLandmarks : public SystemVisualNav
{
public:
    explicit SystemVisualNavPointLandmarks(const GaussianInfo<double> & density);
    SystemVisualNav * clone() const override;
    virtual std::size_t numberLandmarks() const override;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const override;

    // New methods
    void incrementFailedObservations(std::size_t idxLandmark);
    int getFailedObservations(std::size_t idxLandmark) const;
    void resetFailedObservations(std::size_t idxLandmark);
    void removeLandmark(std::size_t idxLandmark);
    void initializeNewLandmark(const Eigen::Vector3d& position);

    private:
        std::map<std::size_t, int> failedObservations_;
};

#endif