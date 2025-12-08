#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <filesystem>

// void calibrateCamera(const std::filesystem::path & configPath);
void calibrateCamera(const std::filesystem::path & configPath, bool exportImages = true, 
                     const std::filesystem::path & outputDirectory = "");


#endif