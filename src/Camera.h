#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"
#include "Pose.hpp"
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>

struct Chessboard
{
    cv::Size boardSize;
    float squareSize;

    void write(cv::FileStorage & fs) const;                 // OpenCV serialisation
    void read(const cv::FileNode & node);                   // OpenCV serialisation

    std::vector<cv::Point3f> gridPoints() const;
    friend std::ostream & operator<<(std::ostream &, const Chessboard &);
};

struct Camera;

struct ChessboardImage
{
    ChessboardImage(const cv::Mat &, const Chessboard &, const std::filesystem::path & = "");
    cv::Mat image;
    std::filesystem::path filename;
    Pose<double> Tnc;                                       // Extrinsic camera parameters
    std::vector<cv::Point2f> corners;                       // Chessboard corners in image [rQOi]
    bool isFound;
    void drawCorners(const Chessboard &);
    void drawBox(const Chessboard &, const Camera &);
    void recoverPose(const Chessboard &, const Camera &);
};

struct ChessboardData
{
    explicit ChessboardData(const std::filesystem::path &); // Load from config file

    Chessboard chessboard;
    std::vector<ChessboardImage> chessboardImages;

    void drawCorners();
    void drawBoxes(const Camera &);
    void recoverPoses(const Camera &);
};

namespace Eigen {
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}

struct Camera
{
    void calibrate(ChessboardData &);                       // Calibrate camera from chessboard data
    void printCalibration() const;

    template <typename Scalar> Pose<Scalar> cameraToBody(const Pose<Scalar> & Tnc) const { return Tnc*Tbc.inverse(); }  // Tnb = Tnc*Tcb
    template <typename Scalar> Pose<Scalar> bodyToCamera(const Pose<Scalar> & Tnb) const { return Tnb*Tbc; }            // Tnc = Tnb*Tbc
    Eigen::Matrix<double, 2, Eigen::Dynamic> undistort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOi) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> undistort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQOi) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> distort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQbarOi) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> distort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQbarOi) const;
    
    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const;
    cv::Vec2d worldToPixel(const cv::Vec3d &, const Pose<double> &) const;
    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> vectorToPixel(const Eigen::Vector3<Scalar> &) const;
    Eigen::Vector2d vectorToPixel(const Eigen::Vector3d &, Eigen::Matrix23d &) const;

    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size

    Pose<double> Tbc;                                       // Relative pose of camera in body coordinates (Rbc, rCBb)
    template <typename Scalar>
    Eigen::Vector2<Scalar> worldToPixel(const Eigen::Vector3<Scalar>& rPNn, const Pose<Scalar>& Tnb) const
    {   
        //std::cout << "rPNn input into worldtopixel: " << rPNn.transpose() << std::endl;
        // Camera pose Tnc (i.e., Rnc, rCNn)
        Pose<Scalar> Tnc = Tnb * Tbc;

        // Compute the vector from the camera to the point in camera coordinates
        Eigen::Vector3<Scalar> rPCc = Tnc.rotationMatrix.transpose() * (rPNn - Tnc.translationVector);

        //std::cout << "rPCc before normalization: " << rPCc.transpose() << std::endl;

        // Normalize the resulting vector
        Eigen::Vector3<Scalar> uPCc = rPCc.normalized();

        //std::cout << "uPCc after normalization: " << uPCc.transpose() << std::endl;

        // Project the normalized vector to pixel coordinates
        Eigen::Vector2<Scalar> pixel = vectorToPixel(uPCc);

        //std::cout << "Projected pixel: " << pixel.transpose() << std::endl;

        return pixel;
    }

    // Overloaded method for Eigen::Vector3<autodiff::dual>
    bool isWorldWithinFOV(const Eigen::Vector3<autodiff::dual>& rPNn, const Pose<autodiff::dual>& Tnb) const
    {
        // Convert Eigen::Vector3<autodiff::dual> to cv::Vec3d
        cv::Vec3d rPNn_cv(autodiff::val(rPNn[0]), autodiff::val(rPNn[1]), autodiff::val(rPNn[2]));

        // Convert Pose<autodiff::dual> to Pose<double>
        Pose<double> Tnb_double;
        Tnb_double.rotationMatrix = Tnb.rotationMatrix.unaryExpr([](const autodiff::dual& x) { return autodiff::val(x); });
        Tnb_double.translationVector = Tnb.translationVector.unaryExpr([](const autodiff::dual& x) { return autodiff::val(x); });

        // Call the original implementation
        return isWorldWithinFOV(rPNn_cv, Tnb_double);
    }


private:
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view
};

#include <cmath>
#include <opencv2/calib3d.hpp>

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::vectorToPixel(const Eigen::Vector3<Scalar> & rPCc) const
{
    // bool isRationalModel    = (flags & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
    // bool isThinPrismModel   = (flags & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
    // assert(isRationalModel && isThinPrismModel);
    //std::cout << "Input vector: " << rPCc.transpose() << std::endl;
    Eigen::Vector2<Scalar> rQOi;
    
    // TODO: Lab 7 (optional)

    // Normalize the input vector
    Scalar x = rPCc[0] / rPCc[2];
    Scalar y = rPCc[1] / rPCc[2];

    // Compute r^2
    Scalar r2 = x*x + y*y;
    Scalar r4 = r2*r2;
    Scalar r6 = r4*r2;

    // Get distortion coefficients
    Scalar k1 = distCoeffs.at<double>(0);
    Scalar k2 = distCoeffs.at<double>(1);
    Scalar p1 = distCoeffs.at<double>(2);
    Scalar p2 = distCoeffs.at<double>(3);
    Scalar k3 = distCoeffs.at<double>(4);
    Scalar k4 = distCoeffs.at<double>(5);
    Scalar k5 = distCoeffs.at<double>(6);
    Scalar k6 = distCoeffs.at<double>(7);
    Scalar s1 = distCoeffs.at<double>(8);
    Scalar s2 = distCoeffs.at<double>(9);
    Scalar s3 = distCoeffs.at<double>(10);
    Scalar s4 = distCoeffs.at<double>(11);

    // Compute radial distortion
    Scalar alpha = Scalar(1) + k1*r2 + k2*r4 + k3*r6;
    Scalar beta = Scalar(1) + k4*r2 + k5*r4 + k6*r6;
    Scalar radial = alpha / beta;

    // Compute tangential distortion
    Scalar dx = Scalar(2)*p1*x*y + p2*(r2 + Scalar(2)*x*x);
    Scalar dy = p1*(r2 + Scalar(2)*y*y) + Scalar(2)*p2*x*y;

    // Compute thin prism distortion
    dx += s1*r2 + s2*r4;
    dy += s3*r2 + s4*r4;

    // Apply distortion
    Scalar xd = x * radial + dx;
    Scalar yd = y * radial + dy;

    // Apply camera matrix
    Scalar fx = cameraMatrix.at<double>(0, 0);
    Scalar fy = cameraMatrix.at<double>(1, 1);
    Scalar cx = cameraMatrix.at<double>(0, 2);
    Scalar cy = cameraMatrix.at<double>(1, 2);
    //std::cout << "After distortion: xd = " << xd << ", yd = " << yd << std::endl;

    rQOi[0] = fx * xd + cx;
    rQOi[1] = fy * yd + cy;
    //std::cout << "Final pixel coordinates: " << rQOi.transpose() << std::endl;
    return rQOi;
}

#endif
