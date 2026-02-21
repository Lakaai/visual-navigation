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
    auto start1 = std::chrono::high_resolution_clock::now();
    system.predict(time_);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::cout << "predict took: " << duration1.count() << " microseconds" << std::endl;

    // Event-specific implementation
    auto start2 = std::chrono::high_resolution_clock::now();
    update(system);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout << "update took: " << duration2.count() << " microseconds" << std::endl;

    if (verbosity_ > 0)
    {
        std::cout << " done" << std::endl;
    }
}

std::string Event::getProcessString() const
{
    return "Processing event:";
}