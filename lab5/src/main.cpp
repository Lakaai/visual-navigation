/**
 * @mainpage MCHA4400 Lab 5: Laplace filtering
 *
 * @tableofcontents
 *
 * @section intro Introduction
 *
 * In this lab, you will:
 * - Implement Gaussian distribution operations in square-root moment form
 * - Implement the process dynamics and measurement model for a ballistic state estimation problem
 * - Run a square-root Laplace filter
 * - Plot the results using VTK
 *
 * @section tasks Tasks
 *
 * 1. Implement Gaussian distribution operations (log likelihood, affine transform, marginal, conditional)
 * 2. Implement ballistic process model dynamics 
 * 3. Implement RADAR range measurement model
 * 4. Run Laplace filter and visualize results
 *
 * @section implementation Key Implementation Files
 * 
 * - GaussianBase.hpp: Base class for Gaussian distribution
 * - Gaussian.hpp: Gaussian distribution in square-root moment form
 * - SystemBallistic.cpp: System dynamics for ballistic trajectory
 * - MeasurementRADAR.cpp: Measurement model and likelihood for RADAR
 * - ballistic_plot.cpp: Plotting functions for ballistic trajectory
 *
 * @section testing Unit Tests
 *
 * Unit tests are provided to verify your implementations:
 * 
 * - GaussianLog.cpp
 * - GaussianTransform.cpp  
 * - GaussianMarginal.cpp
 * - GaussianConditional.cpp
 * - GaussianConfidence.cpp
 * - SystemBallistic.cpp
 * - MeasurementRADAR.cpp
 *
 * @section build Building and Running
 *
 * 1. Configure CMake build:
 *    `cmake -G Ninja -B build -DBUILD_DOCUMENTATION=ON -DCMAKE_BUILD_TYPE=Debug && cd build`
 * 
 * 2. Build project and run unit tests:
 *    `ninja`
 *
 * 3. Build and run executable:
 *    `ninja && ./lab5`
 */
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Core>
#include "Gaussian.hpp"
#include "SystemBallistic.h"
#include "MeasurementRADAR.h"
#include "ballistic_plot.h"

int main(int argc, char *argv[])
{
    std::string fileName   = "../data/estimationdata.csv";
    
    // Dimensions of state and measurement vectors for recording results
    const std::size_t nx = 3;
    const std::size_t ny = 1;

    Eigen::VectorXd x0(nx);
    Eigen::VectorXd u;
    Eigen::VectorXd t_hist;
    Eigen::MatrixXd x_hist, y_hist;

    // Read from CSV
    std::fstream input;
    input.open(fileName, std::fstream::in);
    if (!input.is_open())
    {
        std::cout << "Could not open input file \"" << fileName << "\"! Exiting" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Reading data from " << fileName << std::endl;

    // Determine number of time steps
    std::size_t rows = 0;
    std::string line;
    while (std::getline(input, line))
    {
        rows++;
    }
    std::cout << "Found " << rows << " rows within " << fileName << std::endl << std::endl;
    std::size_t nsteps = rows - 1;  // Disregard header row

    t_hist.resize(nsteps);
    x_hist.resize(nx, nsteps);
    y_hist.resize(ny, nsteps);

    // Read each row of data
    rows = 0;
    input.clear();
    input.seekg(0);
    std::vector<std::string> row;
    std::string csvElement;
    while (std::getline(input, line))
    {
        if (rows > 0)
        {
            std::size_t i = rows - 1;
            
            row.clear();

            std::stringstream s(line);
            while (std::getline(s, csvElement, ','))
            {
                row.push_back(csvElement);
            }
            
            t_hist(i)    = std::stof(row[0]);
            x_hist(0, i) = std::stof(row[1]);
            x_hist(1, i) = std::stof(row[2]);
            x_hist(2, i) = std::stof(row[3]);
            y_hist(0, i) = std::stof(row[5]);
        }
        rows++;
    }

    Eigen::MatrixXd mu_hist(nx, nsteps);
    Eigen::MatrixXd sigma_hist(nx, nsteps);

    // Initial state estimate
    Eigen::MatrixXd S0(nx, nx);
    Eigen::VectorXd mu0(nx);
    S0.fill(0);
    S0.diagonal() << 2200, 100, 1e-3;

    mu0 << 14000, // Initial height
            -450, // Initial velocity
          0.0005; // Ballistic coefficient

    Gaussian<double> p0 = Gaussian<double>::fromSqrtMoment(mu0, S0);
    SystemBallistic system(p0);

    std::cout << "Initial state estimate" << std::endl;
    std::cout << "mu[0] = \n" << p0.mean() << std::endl;
    std::cout << "P[0] = \n" << p0.cov() << std::endl;

    for (std::size_t k = 0; k < nsteps; ++k)
    {
        // Create RADAR measurement
        double t = t_hist(k);
        Eigen::VectorXd y = y_hist.col(k);
        MeasurementRADAR measurementRADAR(t, y);

        // Process measurement event (do time update and measurement update)
        measurementRADAR.process(system);

        // Save results for plotting
        mu_hist.col(k)       = system.density.mean();
        sigma_hist.col(k)    = system.density.cov().diagonal().cwiseSqrt();
    }

    std::cout << std::endl;
    std::cout << "Final state estimate" << std::endl;
    std::cout << "mu[end] = \n" << system.density.mean() << std::endl;
    std::cout << "P[end] = \n" << system.density.cov() << std::endl;

    // Plot results
    plot_simulation(t_hist, x_hist, mu_hist, sigma_hist);

    return EXIT_SUCCESS;
}

