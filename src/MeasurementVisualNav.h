#ifndef MEASUREMENTVISUALNAV_H
#define MEASUREMENTVISUALNAV_H

#include <cstddef>
#include <vector>
#include "Camera.h"
#include "SystemVisualNav.h"
#include "Measurement.h"

class MeasurementVisualNav : public Measurement
{
public:
    MeasurementVisualNav(double time, const Camera & camera);
    virtual MeasurementVisualNav * clone() const = 0;
    virtual ~MeasurementVisualNav();

    virtual GaussianInfo<double> predictFeatureDensity(const SystemVisualNav & system, std::size_t idxLandmark) const = 0;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const = 0;
    virtual const std::vector<int> & associate(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) = 0;

protected:
    const Camera & camera_;
};

#endif