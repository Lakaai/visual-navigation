import cv2
import numpy as np

import navpy
from src.gaussian import Gaussian
from src.srt_parser import parse_srt
from src.system_estimator import SystemEstimator
from src.measurement import Measurement, MeasurementBaro
from src.update_method import UpdateMethod
from src.sensor_type import SensorType
from src.rotations import Rotations
from src.measurement_flow_bundle import MeasurementFlowBundle, CAMERA
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import argparse

def construct_initial_density():
    mu = np.zeros(18)

    h = 63.126999                               # Altitude (GPS) [m]
    ga = h - 7.0                                # Altitude (AGL) [m]
    
    etak = np.array([0.0, 0.0, -ga, -0.009, 0.09, -1.0 ])
    etakm1 = etak.copy()

    mu[0:3] = np.array([0.0, 0.0, 0.0])         # Initial translational velocity (m/s)
    mu[3:6] = np.array([0.0, 0.0, 0.0])         # Initial angular velocity (rad/s)
    mu[6:12] = etak
    mu[12:18] = etakm1
    
    P = np.eye(18)

    P[:3, :3] = np.eye(3) * 0.1                 # Initial translational velocity covariance (m^2/s^2)
    P[3:6, 3:6] = np.eye(3) * 0.01              # Initial angular velocity covariance (rad^2/s^2)
    
    P[6:9, 6:9] = np.eye(3) * 0.001              # Initial position covariance (m^2/s^2)
    P[9:12, 9:12] = np.eye(3) * 0.001            # Initial attitude covariance (rad^2/s^2)

    P[12:15, 12:15] = np.eye(3) * 0.001         # Initial previous position covariance (m^2/s^2)
    P[15:18, 15:18] = np.eye(3) * 0.001         # Initial previous attitude covariance (rad^2/s^2)

    return Gaussian.from_moment(mu, P)

measurement_data = parse_srt("data/outdoor/flight.SRT")

# rr.init("Visual Navigation", spawn=True)

blueprint = rrb.Horizontal(
    rrb.Spatial3DView(origin="/world", name="World"),
    rrb.Spatial2DView(origin="/world/camera", name="Camera", contents=["/world/**"]),
    rrb.Spatial2DView(origin="/image", name="Frame", contents=["/image/**"]), 
)

# Create empty arg parser for rerun
parser = argparse.ArgumentParser(description="")

# rr.script_add_args(parser)

args = parser.parse_args()

# rr.script_setup(args, "Visual Navigation", default_blueprint=blueprint)

def run_visual_navigation():
    cap = cv2.VideoCapture("data/outdoor/flight.MOV")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {width}x{height}, Number of frames: {num_frames}")
    orb = cv2.ORB_create()

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    initial_density = construct_initial_density()

    system = SystemEstimator(initial_density)

    t0 = measurement_data.timestamp[0]

    lat_ref = measurement_data.latitude[0]
    lon_ref = measurement_data.longitude[0]
    alt_ref = 7

    trajectory_enu = []
    estimated_trajectory_enu = []

    previous_frame = None
    previous_alt = 0
    rQOikm1 = None

    for i in range(len(measurement_data.timestamp)):

        # Read frame 
        ret, frame = cap.read()

        if not ret:
            break  # No more frames therefore end of video

        # Add GPS data to the rerun timeline.
        alt = measurement_data.altitude[i]
        lat = measurement_data.latitude[i]
        lon = measurement_data.longitude[i]
        t = measurement_data.timestamp[i] - t0

        print(f"Time: {t}, Altitude: {alt}, Latitude: {lat}, Longitude: {lon}")

        # Convert GPS coordinates to ECEF
        ned = navpy.lla2ned(lat, lon, alt, lat_ref, lon_ref, alt_ref)
        print(f"North: {ned[0]}, East: {ned[1]}, Down: {ned[2]}")

        # Predict system forward and update with the optical flow measurement.
        system.predict(t, UpdateMethod.UNSCENTED)
        
        measurement_flow_bundle = Measurement(time=t, sensor=SensorType.CAMERA, data=MeasurementFlowBundle(time=t, imgk_raw=frame, imgkm1_raw=previous_frame, rQOikm1=rQOikm1))
        
        system.update(measurement_flow_bundle, UpdateMethod.BFGS)

        if alt != previous_alt:
            measurement_baro = Measurement(time=t, sensor=SensorType.BARO, data=MeasurementBaro(alt=np.array([alt])))
            system.update(measurement_baro, UpdateMethod.UNSCENTED)
        
        else:
            print("Altitude is the same for this timestep!")

        cv2.putText(frame, f"Frame: {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Visualise the points
        # if measurement_flow_bundle.data.pk is not None:
            
            # for pt in measurement_flow_bundle.data.pk.T:
            #     # pt is [x, y, z] or [x, y, 1]
            #     x, y = pt[0], pt[1]
        
            #     # Draw using integer coordinates
            #     cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        plot_horizon(frame, system, CAMERA)
        cv2.imshow("Good Points", frame)
        cv2.waitKey(1)

        # rr.set_time(timeline="Timeline", duration=t)

        # # Convert NED to ENU for visualisation
        # trajectory_enu.append(np.array([ned[1], ned[0], -ned[2]]))

        # # Log the camera extrinsics as a transform, this will move the camera frustrum to the correct position in the world
        # rr.log("world/camera", rr.Transform3D(translation=[ned[0], ned[1], -ned[2]],  rotation=rr.Quaternion(xyzw=(0, 0, 0, 1))))

        # # Log a visible point at the camera origin
        # rr.log("world/camera", rr.Points3D([ned[0], ned[1], -ned[2]], colors=np.array([[255, 0, 0]]), radii=0.05))
        # rr.log("world/gps_trajectory", rr.LineStrips3D([np.array(trajectory_enu)], colors=[[0, 255, 0]]))

        # rr.log("/image/frame", rr.Image(frame))
        
        estimated_trajectory_enu.append(np.array([system.density.mean[6], system.density.mean[7], -system.density.mean[8]]))

        # Add state estimate to the rerun timeline.
        # rr.log("world/estimated_trajectory", rr.LineStrips3D([np.array(estimated_trajectory_enu)], colors=[[0, 0, 255]]))

        # 4. Update the previous frame and previous flow measurement for the next iteration.
        previous_frame = frame.copy()
        rQOikm1 = measurement_flow_bundle.data.rQOik.copy()
        
        # rr.log("world/chessboard_corners", rr.Points3D(grid_points, colors=np.array([[0, 255, 0]]), radii=0.0015))

        # rr.log("world/camera/pinhole", rr.Pinhole(image_from_camera=camera_matrix, resolution=(width, height), camera_xyz=rr.ViewCoordinates.RDF))
        
        # _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        # rr.log("/image/frame", rr.EncodedImage())
        
        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        # corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # # annotated_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        # # cv2.imshow("Features", annotated_frame)
        # corners = np.int32(corners)

    cap.release()
    cv2.destroyAllWindows()

# def plot_horizon(frame, system, camera, divisor=1):

#     state = system.density.mean

#     # Assuming:
#     # position = state[6:9]
#     # roll pitch yaw = state[9:12]
#     rpy = state[9:12]

#     # Your rotation function
#     Rnb = Rotations.rpy2rot(rpy).R

#     print(Rnb)

#     print(CAMERA.rotation_matrix)
#     # Body-to-camera rotation
#     Rbc = CAMERA.rotation_matrix.R

#     # Navigation-to-camera
#     Rnc = Rnb @ Rbc

#     num_points = 200

#     horizon_points = []

#     for i in range(num_points):

#         angle = 2.0 * np.pi * i / num_points

#         # Direction vector on horizontal plane
#         dir_world = np.array([
#             np.cos(angle),
#             np.sin(angle),
#             0.0
#         ])

#         # Transform into camera frame
#         dir_cam = Rnc.T @ dir_world

#         # Only project points in front of camera
#         if dir_cam[2] <= 0:
#             continue

#         # Project using pinhole model
#         pixel = camera.vector_to_pixel(dir_cam)

#         if pixel is None:
#             continue

#         u, v = pixel

#         # Check image bounds
#         h, w = frame.shape[:2]

#         if 0 <= u < w and 0 <= v < h:
#             horizon_points.append(
#                 (int(u / divisor), int(v / divisor))
#             )

#     # Draw connected horizon
#     if len(horizon_points) > 1:

#         for i in range(len(horizon_points) - 1):

#             cv2.line(
#                 frame,
#                 horizon_points[i],
#                 horizon_points[i + 1],
#                 (0, 0, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#         cv2.line(
#             frame,
#             horizon_points[-1],
#             horizon_points[0],
#             (0, 0, 255),
#             2,
#             cv2.LINE_AA
#         )

import numpy as np
import cv2


def plot_horizon(frame, system, camera):

    state = system.density.mean

    #
    # Extract roll/pitch/yaw
    #
    rpy = state[9:12]

    #
    # Navigation -> body
    #
    Rnb = Rotations.rpy2rot(rpy).R

    #
    # Body -> camera
    #
    Rbc = CAMERA.rotation_matrix.R

    #
    # Navigation -> camera
    #
    Rnc = Rnb @ Rbc

    #
    # Camera intrinsics
    #
    K = CAMERA.matrix

    #
    # Horizon line:
    #
    # l ∝ K^{-T} R_cn [0 0 1]^T
    #
    # line = [a b c]
    # ax + by + c = 0
    #
    n_world = np.array([0.0, 0.0, 1.0])

    line = np.linalg.inv(K).T @ (Rnc @ n_world)

    a, b, c = line

    #
    # Avoid divide-by-zero
    #
    eps = 1e-8

    h, w = frame.shape[:2]

    points = []

    #
    # Intersect with left/right image borders
    #

    # x = 0
    if abs(b) > eps:
        y0 = -(a * 0 + c) / b

        if 0 <= y0 < h:
            points.append((0, int(y0)))

    # x = w-1
    if abs(b) > eps:
        y1 = -(a * (w - 1) + c) / b

        if 0 <= y1 < h:
            points.append((w - 1, int(y1)))

    #
    # Intersect with top/bottom borders
    #

    # y = 0
    if abs(a) > eps:
        x0 = -(b * 0 + c) / a

        if 0 <= x0 < w:
            points.append((int(x0), 0))

    # y = h-1
    if abs(a) > eps:
        x1 = -(b * (h - 1) + c) / a

        if 0 <= x1 < w:
            points.append((int(x1), h - 1))

    #
    # Remove duplicates
    #
    unique_points = []

    for p in points:
        if p not in unique_points:
            unique_points.append(p)

    #
    # Draw line if we found two intersections
    #
    if len(unique_points) >= 2:

        cv2.line(
            frame,
            unique_points[0],
            unique_points[1],
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

run_visual_navigation()
