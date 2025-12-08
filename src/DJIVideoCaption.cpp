#include <cassert>
#include <cstdio>
#include <string>
#include <filesystem>
#include <stdexcept>
#include "DJIVideoCaption.h"

// std::vector<DJIVideoCaption> getVideoCaptions(const std::filesystem::path & captionPath)
// {
//     assert(std::filesystem::exists(captionPath));

//     std::vector<DJIVideoCaption> caps;
//     FILE* file = std::fopen(captionPath.string().c_str(), "r");
//     if (!file) {
//         throw std::runtime_error("Unable to open file: " + captionPath.string());
//     }

//     DJIVideoCaption cap;
//     int hours, minutes, seconds, milliseconds;
//     char buffer[256];

//     while (true) {
//         // Read frame number
//         if (std::fscanf(file, "%d\n", &cap.frameNum) != 1) break;

//         // Read timestamp
//         if (std::fscanf(file, "%d:%d:%d,%d --> %*d:%*d:%*d,%*d\n", 
//                         &hours, &minutes, &seconds, &milliseconds) != 4) break;
//         cap.time = hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0;

//         // Read and parse the data line
//         if (std::fgets(buffer, sizeof(buffer), file) == nullptr) break;
//         if (std::sscanf(buffer, 
//                         "[iso : %d] [shutter : 1/%lf] [fnum : %lf] "
//                         "[lat : %lf] [lon : %lf] [altitude : %lf]",
//                         &cap.iso, &cap.shutterHz, &cap.fnum,
//                         &cap.latitude, &cap.longitude, &cap.altitude) != 6) break;

//         caps.push_back(cap);

//         // Read the empty line between entries
//         if (std::fgets(buffer, sizeof(buffer), file) == nullptr) break;
//     }

//     std::fclose(file);
//     return caps;
// }
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>

std::vector<DJIVideoCaption> getVideoCaptions(const std::filesystem::path & captionPath)
{
    assert(std::filesystem::exists(captionPath));

    std::vector<DJIVideoCaption> caps;
    FILE* file = std::fopen(captionPath.string().c_str(), "r");
    if (!file) {
        throw std::runtime_error("Unable to open file: " + captionPath.string());
    }

    DJIVideoCaption cap;
    int hours, minutes, seconds, milliseconds;
    char buffer[1024];
    int lineCount = 0;
    int captionCount = 0;

    while (true) {
        // Read frame number
        if (std::fscanf(file, "%d\n", &cap.frameNum) != 1) {
            if (feof(file)) break;
            std::cout << "Failed to read frame number at line " << lineCount << std::endl;
            break;
        }
        lineCount++;

        // Read timestamp
        if (std::fscanf(file, "%d:%d:%d,%d --> %*d:%*d:%*d,%*d\n", 
                        &hours, &minutes, &seconds, &milliseconds) != 4) {
            std::cout << "Failed to read timestamp at line " << lineCount << std::endl;
            break;
        }
        cap.time = hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0;
        lineCount++;

        // Read the <font> tag line
        if (std::fgets(buffer, sizeof(buffer), file) == nullptr) {
            std::cout << "Failed to read <font> tag at line " << lineCount << std::endl;
            break;
        }
        lineCount++;

        // Read the FrameCnt line
        if (std::fgets(buffer, sizeof(buffer), file) == nullptr) {
            std::cout << "Failed to read FrameCnt line at line " << lineCount << std::endl;
            break;
        }
        lineCount++;

        // Read and parse the data line
        if (std::fgets(buffer, sizeof(buffer), file) == nullptr) {
            std::cout << "Failed to read data line at line " << lineCount << std::endl;
            break;
        }
        lineCount++;

        int dummy_ev, dummy_ct, dummy_focal_len;
        char dummy_color_md[8];
        double shutterSpeed;
        if (std::sscanf(buffer, 
                        "[iso : %d] [shutter : 1/%lf] [fnum : %lf] [ev : %d] [ct : %d] [color_md : %7s] [focal_len : %d] [latitude : %lf] [longtitude : %lf] [altitude: %lf]",
                        &cap.iso, &shutterSpeed, &cap.fnum,
                        &dummy_ev, &dummy_ct, dummy_color_md, &dummy_focal_len,
                        &cap.latitude, &cap.longitude, &cap.altitude) != 10) {
            std::cout << "Failed to parse data at line " << lineCount << ": " << buffer << std::endl;
            break;
        }
        cap.shutterHz = shutterSpeed;  // Store the actual frequency
        cap.fnum /= 100.0;  // Divide fnum by 100

        caps.push_back(cap);
        captionCount++;

        // Read and discard any remaining characters until the next newline
        int c;
        while ((c = fgetc(file)) != EOF && c != '\n');
        if (c == EOF) break;
        lineCount++;

    }

    std::fclose(file);
    
    return caps;
}