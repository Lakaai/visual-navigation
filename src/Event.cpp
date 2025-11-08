#include <iostream>
#include <format>
#include <string>
#include "SystemBase.h"
#include "Event.h"
#include "visualNavigation.h"

Event::Event(double time)
    : time_(time)
    , verbosity_(1)
{}

Event::Event(double time, int verbosity)
    : time_(time)
    , verbosity_(verbosity)
{}

Event::~Event() = default;

void Event::process(SystemBase & system, int scenario)
{
    if (verbosity_ > 0)
    {
        std::cout << std::format("[t={:07.3f}s] {}", time_, getProcessString());
    }

    // Time update
    system.predict(time_, scenario);
    
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