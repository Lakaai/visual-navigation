#include <doctest/doctest.h>
#include <filesystem>
#include <cmath>
#include <iostream>
#include "../../src/DJIVideoCaption.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

SCENARIO("DJIVideoCaption: Check first caption")
{
    GIVEN("Subtitle file")
    {
        //std::filesystem::path captionPath = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("flight.SRT");
        std::filesystem::path captionPath = "/home/luke/MCHA4400/a2/data/outdoor/flight.SRT";

        REQUIRE(std::filesystem::exists(captionPath));

        WHEN("Calling getVideoCaptions")
        {
            std::vector<DJIVideoCaption> djiVideoCaption = getVideoCaptions(captionPath);

            THEN("The correct number of caption frames are found")
            {
                REQUIRE(djiVideoCaption.size() == 9160);

                AND_THEN("frameNum is correct")
                {
                    REQUIRE(djiVideoCaption[0].frameNum == 1);
                    REQUIRE(djiVideoCaption[1].frameNum == 2);
                    REQUIRE(djiVideoCaption[9160-1].frameNum == 9160);
                }

                AND_THEN("time is correct")
                {
                    REQUIRE(djiVideoCaption[0].time == doctest::Approx(0));
                    REQUIRE(djiVideoCaption[1].time == doctest::Approx(0.016));
                    REQUIRE(djiVideoCaption[9160-1].time == doctest::Approx(152.802));
                }

                AND_THEN("iso is correct")
                {
                    REQUIRE(djiVideoCaption[0].iso == 110);
                    REQUIRE(djiVideoCaption[1].iso == 110);
                    REQUIRE(djiVideoCaption[9160-1].iso == 100);
                }

                AND_THEN("shutterHz is correct")
                {
                    REQUIRE(djiVideoCaption[0].shutterHz == doctest::Approx(400));
                    REQUIRE(djiVideoCaption[1].shutterHz == doctest::Approx(400));
                    REQUIRE(djiVideoCaption[9160-1].shutterHz == doctest::Approx(400));
                }

                AND_THEN("fnum is correct")
                {
                    REQUIRE(djiVideoCaption[0].fnum == doctest::Approx(2.80));
                    REQUIRE(djiVideoCaption[1].fnum == doctest::Approx(2.80));
                    REQUIRE(djiVideoCaption[9160-1].fnum == doctest::Approx(2.80));
                }

                AND_THEN("latitude is correct")
                {
                    REQUIRE(djiVideoCaption[0].latitude == doctest::Approx(-32.868837));
                    REQUIRE(djiVideoCaption[1].latitude == doctest::Approx(-32.868837));
                    REQUIRE(djiVideoCaption[9160-1].latitude == doctest::Approx(-32.862855));
                }

                AND_THEN("longitude is correct")
                {
                    REQUIRE(djiVideoCaption[0].longitude == doctest::Approx(151.684716));
                    REQUIRE(djiVideoCaption[1].longitude == doctest::Approx(151.684716));
                    REQUIRE(djiVideoCaption[9160-1].longitude == doctest::Approx(151.680047));
                }

                AND_THEN("altitude is correct")
                {
                    REQUIRE(djiVideoCaption[0].altitude == doctest::Approx(63.126999));
                    REQUIRE(djiVideoCaption[1].altitude == doctest::Approx(63.126999));
                    REQUIRE(djiVideoCaption[9160-1].altitude == doctest::Approx(127.595001));
                }
            }
        }
    }
}
