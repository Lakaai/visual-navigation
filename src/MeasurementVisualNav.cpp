#include "Camera.h"
#include "MeasurementVisualNav.h"

MeasurementVisualNav::MeasurementVisualNav(double time, const Camera & camera)
    : Measurement(time)
    , camera_(camera)
{}

MeasurementVisualNav::~MeasurementVisualNav() = default;