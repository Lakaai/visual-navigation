#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

SystemEstimator::SystemEstimator(const GaussianInfo<double> & density)
    : SystemBase()
    , density(density)
    
{}

SystemEstimator::~SystemEstimator() = default;
void SystemEstimator::predict(double time)
{
    double dt = time - time_;
    
    assert(dt >= 0);

    if (dt == 0.0) {
        return;
    }
    
    auto pdw = processNoiseDensity(dt);     
    auto pxdw = density*pdw;                

    auto func = [&](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J) { 
        const int nx = 12;  
        const int nz = 6;   
        const int nw = 6;   
        const int nl = (xdw.size() - nx - nw - nz)/3;  

        J = Eigen::MatrixXd::Zero(nx + nz + 3*nl, nx + nz + 3*nl + nw);

        Eigen::MatrixXd Jz; 
        Eigen::VectorXd z = xdw.head(12);                
        Eigen::VectorXd dw = xdw.tail(6);                
        Eigen::VectorXd eta = xdw.segment(6, 6);     
        Eigen::VectorXd zeta = xdw.segment(12, 6);       // Get current zeta    
        Eigen::VectorXd m = xdw.segment(18, 3*nl);       
        
        Eigen::VectorXd zdw(z.size() + dw.size());
        zdw << z, dw;  

        //  std::cout << "\nPrediction step values:" << std::endl;
        // std::cout << "eta: " << eta.transpose() << std::endl;
        // std::cout << "current zeta: " << zeta.transpose() << std::endl;

        // Choose which eta to use based on update_zeta_ flag
        Eigen::VectorXd zetakp1;
        if (update_zeta_) {     
            zetakp1 = eta;  
}        else {
            zetakp1 = zeta;                                       
            //std::cout << "zetakp1 = current zeta: " << zetakp1.transpose() << std::endl;
        }
        
        Eigen::VectorXd xkp1(RK4SDEHelper(zdw, dt, Jz).size() + eta.size() + m.size());
        xkp1 << RK4SDEHelper(zdw, dt, Jz), zetakp1, m;
        // std::cout << "update_zeta_ flag: " << update_zeta_ << std::endl;
        // std::cout << "  eta segment: " << xkp1.segment(6,6).transpose() << std::endl;
        // std::cout << "  zeta segment: " << xkp1.segment(12,6).transpose() << std::endl;


        // Set up Jacobian based on update behavior
        J.topLeftCorner(nx, nx) = Jz.topLeftCorner(nx, nx);
        J.topRightCorner(nx, nw) = Jz.topRightCorner(nx, nw);
        if (update_zeta_) {
            J.block(nx, 6, 6, 6) = Eigen::MatrixXd::Identity(6, 6);
        }
        else {
            J.block(nx, nx, 6, 6) = Eigen::MatrixXd::Identity(6, 6);
        }
        
        J.block(nx + nz, nx + nx, 3*nl, 3*nl) = Eigen::MatrixXd::Identity(3*nl, 3*nl);
        return xkp1;
    };

    //std::cout << "update_zeta_ flag: " << update_zeta_ << std::endl;
    //std::cout << "zeta density before affine transform: " << density.mean().segment<6>(12) << std::endl;
    //Eigen::VectorXd zeta_density_before = density.mean().segment<6>(12);
    density = pxdw.affineTransform(func);
    // if (update_zeta_) {     
    //         //std::cout << "zeta updated therefore state left alone" << std::endl;
      
    // } else {
    //         density.mean().segment<6>(12) = zeta_density_before;                                    
    //         //std::cout << "zetakp1 = current zeta: " << zetakp1.transpose() << std::endl;
    // }
    
    //std::cout << "zeta density after affine transform: " << density.mean().segment<6>(12) << std::endl;
    
    time_ = time;
}

// void SystemEstimator::predict(double time)
// {
//     double dt = time - time_;
//     std::cout << "dt= \n" << dt << std::endl;
//     assert(dt >= 0);

//     if (dt == 0.0) {
//             return;
//     }
    
//     // Augment state density with independent noise increment dw ~ N^{-1}(0, LambdaQ/dt)
//     // [ x] ~ N^{-1}([ eta ], [ Lambda,          0 ])
//     // [dw]         ([   0 ]  [      0, LambdaQ/dt ])
//     auto pdw = processNoiseDensity(dt);     // p(dw(idxQ)[k])
//     // Print zdw to verify the result
//         // std::cout << "pdw: " << pdw.mean() << std::endl;

//     auto pxdw = density*pdw;                // p(x[k], dw(idxQ)[k]) = p(x[k])*p(dw(idxQ)[k])
    
//     // xdw now becomes zdw 
//     // Map [ x[k]; dw(idxQ)[k] ] to x[k+1] // Func is phi 
//     auto func = [&](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J) { 
//         // Determine the sizes
//         const int nx = 12;  // Assuming 12 state variables
//         const int nz = 6;   // Assuming 6 zeta variables
//         const int nw = 6;   // Assuming 6 noise variables
//         const int nl = (xdw.size() - nx - nw - nz)/3;  // number of landmarks

//         // Initialise the full Jacobian of the phi map
//         J = Eigen::MatrixXd::Zero(nx + nz + 3*nl, nx + nz + 3*nl + nw);

//         // Initialise the original Jacobian for Jz [dx[k+1]/dx[k], dx[k+1]/dw[k]]
//         Eigen::MatrixXd Jz; 

//         Eigen::VectorXd z   = xdw.head(12);                 // Extract first 12 elements (indices 0-11) of xdw into z
//         Eigen::VectorXd dw  = xdw.tail(6);                  // Extract elements 12 to 17 and store in zdw
//         Eigen::VectorXd eta = xdw.segment(6, 6);            // Extract eta from xdw 
//         Eigen::VectorXd m   = xdw.segment(18, 3*nl);        // Extract landmark positions from xdw 
        
//         Eigen::VectorXd zdw(z.size() + dw.size());
//         zdw << z, dw;  
        
//         Eigen::VectorXd xkp1(RK4SDEHelper(zdw, dt, Jz).size() + eta.size() + m.size()); // Stack the result of RK4SDEHelper(zdw, dt, Jz) directly with eta
//         xkp1 << RK4SDEHelper(zdw, dt, Jz), eta, m;        

//         J.topLeftCorner(nx, nx) = Jz.topLeftCorner(nx, nx);                             // ∂z[k+1]/∂z[k]
//         J.topRightCorner(nx, nw) = Jz.topRightCorner(nx, nw);                           // ∂z[k+1]/∂w[k]
//         J.block(nx, 6, 6, 6)= Eigen::MatrixXd::Identity(6, 6);                          // ∂zeta[k+1]/∂eta[k]
//         J.block(nx + nz, nx + nx, 3*nl, 3*nl) = Eigen::MatrixXd::Identity(3*nl, 3*nl);  // ∂m[k+1]/∂m[k]

//         // std::cout << "Jz= \n" << Jz << std::endl;

//         // std::cout << "J= \n" << J << std::endl;

//         return xkp1;                                                            // return x[k+1]; 
//         };

    // Map p(z[k], dw(idxQ)[k]) to p(z[k+1])
//     density = pxdw.affineTransform(func);

//     time_ = time;
// }

Eigen::VectorXd SystemEstimator::dynamicsEst(const Eigen::VectorXd & x) const
{
    return dynamics(x);
}

Eigen::VectorXd SystemEstimator::dynamicsEst(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    return dynamics(x, J);
}

// Evaluate F(X) from dX = F(X)*dt + dW
Eigen::MatrixXd SystemEstimator::augmentedDynamicsEst(const Eigen::MatrixXd & X) const
{
    assert(X.size() > 0);
    int nx = X.rows();
    assert(X.cols() == 2*nx + 1);

    Eigen::VectorXd x = X.col(0);
    Eigen::MatrixXd J;
    Eigen::VectorXd f = dynamicsEst(x, J);
    assert(f.rows() == nx);
    assert(J.rows() == nx);
    assert(J.cols() == nx);

    Eigen::MatrixXd dX(nx, 2*nx + 1);
    dX << f, J*X.block(0, 1, nx, 2*nx);
    assert(!dX.hasNaN() && "NaN found in dX inside augmentedDynamicsEst.");
    return dX;
}

// Map [x[k]; dw(idxQ)[k]] to x[k+1] using RK4
Eigen::VectorXd SystemEstimator::RK4SDEHelper(const Eigen::VectorXd & xdw, double dt, Eigen::MatrixXd & J) const
{
    const std::vector<Eigen::Index> & idxQ = processNoiseIndex();

    const std::size_t nx = 12;
    const std::size_t nq = idxQ.size();
    // assert(xdw.size() == nx + nq);

    assert(static_cast<std::size_t>(xdw.size()) == nx + nq);

    Eigen::VectorXd x(nx), dw(nx);
    x = xdw.head(nx);
    dw.setZero();
    dw(idxQ) = xdw.tail(nq);

    typedef Eigen::MatrixXd Matrix;

    // X  = [ x,  dx/dx[k],   dx/dw[k] ]
    // dW = [dw, ddw/dx[k], ddw/ddw[k] ]
    Matrix X(nx, 2*nx+1), dW(nx, 2*nx+1);
    X  <<  x, Matrix::Identity(nx, nx), Matrix::Zero(nx, nx);
    dW << dw,     Matrix::Zero(nx, nx), Matrix::Identity(nx, nx);

    Matrix F1, F2, F3, F4, Xnext;
    F1 = augmentedDynamicsEst(                 X);
    assert(!F1.hasNaN() && "NaN found in F1 inside RK$SDEHelper.");
    F2 = augmentedDynamicsEst(X + (F1*dt + dW)/2);
    assert(!F2.hasNaN() && "NaN found in F2 inside RK$SDEHelper.");
    F3 = augmentedDynamicsEst(X + (F2*dt + dW)/2);
    assert(!F3.hasNaN() && "NaN found in F3 inside RK$SDEHelper.");
    F4 = augmentedDynamicsEst(    X + F3*dt + dW);
    assert(!F4.hasNaN() && "NaN found in F4 inside RK$SDEHelper.");
    Xnext = X + (F1 + 2*F2 + 2*F3 + F4)*dt/6 + dW;
    assert(!Xnext.hasNaN() && "NaN found in Xnext inside RK4SDEHelper.");

    // X[k+1] = [ x[k+1], dx[k+1]/dx[k], dx[k+1]/dw[k] ]
    J.resize(nx, nx + nq);

        // Isolate and check NaNs in the specific sections of Xnext
    // 1. Xnext.middleCols(1, nx)
    if (Xnext.middleCols(1, nx).hasNaN()) {
        std::cout << "NaN found in Xnext.middleCols(1, nx)!" << std::endl;
    }

    // 2. Xnext.middleCols(nx + 1, nx) (the part corresponding to noise Jacobian)
    if (Xnext.middleCols(nx + 1, nx).hasNaN()) {
        std::cout << "NaN found in Xnext.middleCols(nx + 1, nx)!" << std::endl;
    }

    // 3. Extracting specific columns from the noise Jacobian using idxQ
    if (Xnext.middleCols(nx + 1, nx)(Eigen::all, idxQ).hasNaN()) {
        std::cout << "NaN found in Xnext.middleCols(nx + 1, nx)(Eigen::all, idxQ)!" << std::endl;
    }

    J << Xnext.middleCols(1, nx), Xnext.middleCols(nx + 1, nx)(Eigen::all, idxQ);
    assert(!J.hasNaN() && "NaN found in Jacobian matrix inside RK4SDEHelper.");
    return Xnext.col(0);
}
