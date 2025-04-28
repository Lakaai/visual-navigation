#include <doctest/doctest.h>
#include <filesystem>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include "../../src/Pose.hpp"
#include "../../src/Camera.h"

SCENARIO("Camera model")
{
    GIVEN("A camera with no lens distortion")
    {
        
        std::filesystem::path cameraPath("test/data/camera.xml");
        REQUIRE(std::filesystem::exists(cameraPath));
        REQUIRE(std::filesystem::is_regular_file(cameraPath));

        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        REQUIRE(fs.isOpened());

        Camera camera;
        fs["camera"] >> camera;

        REQUIRE(camera.cameraMatrix.cols == 3);
        REQUIRE(camera.cameraMatrix.rows == 3);
        REQUIRE(camera.cameraMatrix.type() == CV_64F);
        REQUIRE(camera.distCoeffs.cols == 1);
        REQUIRE(camera.distCoeffs.type() == CV_64F);

        GIVEN("The positive optical axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 0.0, 1.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = camera.isVectorWithinFOV(uPCc);
                THEN("Vector is within field of view")
                {
                    CHECK(isWithinFOV);
                }
            }

            WHEN("Calling vectorToPixel")
            {
                cv::Vec2d rQOi = camera.vectorToPixel(uPCc);

                THEN("Vector maps to centre of image")
                {
                    const double & cx = camera.cameraMatrix.at<double>(0, 2);
                    const double & cy = camera.cameraMatrix.at<double>(1, 2);

                    CHECK(rQOi(0) == doctest::Approx(cx));
                    CHECK(rQOi(1) == doctest::Approx(cy));
                }
            }
        }

        GIVEN("The negative optical axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 0.0, -1.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = camera.isVectorWithinFOV(uPCc);

                THEN("Vector is outside field of view")
                {
                    CHECK_FALSE(isWithinFOV);
                }
            }
        }

        GIVEN("The positive horizontal image axis unit vector")
        {
            cv::Vec3d uPCc(1.0, 0.0, 0.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = camera.isVectorWithinFOV(uPCc);

                THEN("Vector is outside field of view")
                {
                    CHECK_FALSE(isWithinFOV);
                }
            }
        }

        GIVEN("The positive vertical image axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 1.0, 0.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = camera.isVectorWithinFOV(uPCc);

                THEN("Vector is outside field of view")
                {
                    CHECK_FALSE(isWithinFOV);
                }
            }
        }

        GIVEN("A pixel location")
        {
            cv::Vec2d rQOi_given(5.3, 8.7);
            WHEN("Evaluating pixelToVector")
            {
                cv::Vec3d uPCc = camera.pixelToVector(rQOi_given);
                REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

                THEN("Vector corresponds to given pixel")
                {
                    cv::Vec2d rQOi_actual = camera.vectorToPixel(uPCc);
                    CHECK(rQOi_actual(0) == doctest::Approx(rQOi_given(0)));
                    CHECK(rQOi_actual(1) == doctest::Approx(rQOi_given(1)));
                }
            }
        }

        GIVEN("An arbitrary world point and body pose")
        {
            cv::Vec3d rPNn(0.141886338627215, 0.421761282626275, 0.915735525189067);

            Pose<double> Tnb;
            Tnb.translationVector << 0.792207329559554, 0.959492426392903, 0.655740699156587;
            Tnb.rotationMatrix <<
                 0.988707451899469, -0.0261040852852265, 0.147567446579119,
               -0.0407096903396656,   0.900896143585899, 0.432121348216568,
                -0.144223076069359,  -0.433249022161017,  0.88966004132231;
            REQUIRE((Tnb.rotationMatrix.transpose()*Tnb.rotationMatrix - Eigen::Matrix3d::Identity()).lpNorm<Eigen::Infinity>() < 100*std::numeric_limits<double>::epsilon()); // Must be orthogonal

            WHEN("Evaluating worldToVector")
            {
                cv::Vec3d uPCc = camera.worldToVector(rPNn, Tnb);

                THEN("A unit vector is returned")
                {
                    REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

                    AND_THEN("The expected oracle result is returned")
                    {
                        cv::Vec3d uPCc_oracle(-0.745857120799272, -0.656980343862456, -0.109881677869373);
                        CHECK(uPCc(0) == doctest::Approx(uPCc_oracle(0)));
                        CHECK(uPCc(1) == doctest::Approx(uPCc_oracle(1)));
                        CHECK(uPCc(2) == doctest::Approx(uPCc_oracle(2)));
                    }
                }
            }

            WHEN("Evaluating worldToPixel")
            {
                cv::Vec2d rQOi = camera.worldToPixel(rPNn, Tnb);

                THEN("worldToPixel matches the composition of worldToVector and vectorToPixel")
                {
                    cv::Vec2d rQOi_expected = camera.vectorToPixel(camera.worldToVector(rPNn, Tnb));
                    REQUIRE(rQOi(0) == doctest::Approx(rQOi_expected(0)));
                    REQUIRE(rQOi(1) == doctest::Approx(rQOi_expected(1)));

                    AND_THEN("The expected oracle result is returned")
                    {
                        cv::Vec2d rQOi_oracle(34.039103190885967, 36.073879427476669);
                        CHECK(rQOi(0) == doctest::Approx(rQOi_oracle(0)));
                        CHECK(rQOi(1) == doctest::Approx(rQOi_oracle(1)));
                    }
                }
            }
        }
    }
}

SCENARIO("Lens distortion")
{
    GIVEN("A camera with lens distortion")
    {
        const std::filesystem::path cameraPath = "test/data/camera2.xml";
        REQUIRE(std::filesystem::exists(cameraPath));

        // Load camera calibration
        Camera camera;
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        REQUIRE(fs.isOpened());
        fs["camera"] >> camera;

        REQUIRE(camera.cameraMatrix.rows == 3);
        REQUIRE(camera.cameraMatrix.cols == 3);
        REQUIRE(camera.cameraMatrix.type() == CV_64F);
        REQUIRE(camera.distCoeffs.cols == 1);
        REQUIRE(camera.distCoeffs.type() == CV_64F);

        GIVEN("A test pixel rQOi")
        {
            Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi(2, 1);
            rQOi << 0.25*camera.imageSize.width, 0.25*camera.imageSize.height;

            WHEN("rQbarOi = camera.undistort(rQOi)")
            {
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi = camera.undistort(rQOi);

                THEN("rQbarOi has the correct pixel coordinates")
                {
                    CHECK(std::abs(rQbarOi(0, 0) - 393.0) < 0.1);
                    CHECK(std::abs(rQbarOi(1, 0) - 294.8) < 0.1);
                }

                AND_WHEN("rQOi = camera.distort(rQbarOi)")
                {
                    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi_actual = camera.distort(rQbarOi);

                    THEN("Camera::distort is the inverse of Camera::undistort")
                    {
                        CHECK(std::abs(rQOi_actual(0, 0) - rQOi(0, 0)) < 0.07);
                        CHECK(std::abs(rQOi_actual(1, 0) - rQOi(1, 0)) < 0.07);
                    }
                }
            }

            WHEN("pQbarOi = camera.undistort(pQOi)")
            {
                Eigen::Matrix<double, 3, Eigen::Dynamic> pQOi(3, 1);
                pQOi << rQOi(0, 0), rQOi(1, 0), 1.0;
                Eigen::Matrix<double, 3, Eigen::Dynamic> pQbarOi = camera.undistort(pQOi);

                THEN("pQbarOi has the correct pixel coordinates")
                {
                    CHECK(std::abs(pQbarOi(0, 0) / pQbarOi(2, 0) - 393.0) < 0.1);
                    CHECK(std::abs(pQbarOi(1, 0) / pQbarOi(2, 0) - 294.8) < 0.1);
                }

                AND_WHEN("pQOi = camera.distort(pQbarOi)")
                {
                    Eigen::Matrix<double, 3, Eigen::Dynamic> pQOi_actual = camera.distort(pQbarOi);

                    THEN("Camera::distort is the inverse of Camera::undistort")
                    {
                        CHECK(std::abs(pQOi_actual(0, 0) / pQOi_actual(2, 0) - pQOi(0, 0) / pQOi(2, 0)) < 0.07);
                        CHECK(std::abs(pQOi_actual(1, 0) / pQOi_actual(2, 0) - pQOi(1, 0) / pQOi(2, 0)) < 0.07);
                    }
                }
            }
        }
    }
}
