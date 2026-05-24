#include <iostream>
#include <format>
#include <string>
#include "SystemBase.h"
#include "Event.h"
#include <chrono>

Event::Event(double time)
    : time_(time)
    , verbosity_(0)
{}

Event::Event(double time, int verbosity)
    : time_(time)
    , verbosity_(verbosity)
{}

Event::~Event() = default;

// In Event.cpp - initialize the static member and modify process
// double Event::last_prediction_time_ = -1;  // Initialize to invalid time

void Event::process(SystemBase & system)
{
    if (verbosity_ > 0)
    {
        std::cout << std::format("[t={:07.3f}s] {}", time_, getProcessString());
    }
    
    // Time update
    system.predict(time_);

    // Event-specific implementation
    
    update(system);


    if (verbosity_ > 0)
    {
        std::cout << " done" << std::endl;
    }
}

std::string Event::getProcessString() const
{
    return "Processing event:";
}