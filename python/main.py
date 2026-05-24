import cv2
import numpy as np

import navpy
from src.gaussian import Gaussian
from src.srt_parser import parse_srt
from src.system_estimator import SystemEstimator
from src.measurement import Measurement, MeasurementBaro
from src.rerun_helpers import RerunHelper
from src.update_method import UpdateMethod
from src.sensor_type import SensorType
from src.rotations import Rotations
from src.measurement_flow_bundle import MeasurementFlowBundle
from src.camera import CAMERA
import rerun as rr
import rerun.blueprint as rrb
import argparse
import time

def construct_initial_density():
    mu = np.zeros(18)

    h = 63.126999                               # Altitude (GPS) [m]
    ga = h - 7.0                                # Altitude (AGL) [m]
    
    etak = np.array([0.0, 0.0, -ga, -0.009, np.deg2rad(-5), 0])
    etakm1 = etak.copy()

    mu[0:3] = np.array([0.0, 0.0, 0.0])         # Initial translational velocity (m/s)
    mu[3:6] = np.array([0.0, 0.0, 0.0])         # Initial angular velocity (rad/s)
    mu[6:12] = etak
    mu[12:18] = etakm1
    
    P = np.eye(18)

    P[:3, :3] = np.eye(3) * 0.1                 # Initial translational velocity covariance (m^2/s^2)
    P[3:6, 3:6] = np.eye(3) * 0.01              # Initial angular velocity covariance (rad^2/s^2)
    
    P[6:9, 6:9] = np.eye(3) * 0.001             # Initial position covariance (m^2/s^2)
    P[9:12, 9:12] = np.eye(3) * 0.001           # Initial attitude covariance (rad^2/s^2)

    P[12:15, 12:15] = np.eye(3) * 0.001         # Initial previous position covariance (m^2/s^2)
    P[15:18, 15:18] = np.eye(3) * 0.001         # Initial previous attitude covariance (rad^2/s^2)

    return Gaussian.from_moment(mu, P)

measurement_data = parse_srt("data/outdoor/flight.SRT")

use_rerun = True

if use_rerun:

    rr.init("Visual Navigation", spawn=True)

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="World"),
        rrb.Spatial2DView(origin="/world/camera", name="Camera"),
        # rrb.Spatial2DView(origin="/image", name="Frame", contents=["/image/**"]), 
    )

    # Create empty arg parser for rerun
    parser = argparse.ArgumentParser(description="")

    rr.script_add_args(parser)

    args = parser.parse_args()

    rr.script_setup(args, "Visual Navigation", default_blueprint=blueprint)

    # Intialise world frame as NED (Right-Handed Z-Down)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rerun_helper = RerunHelper()
    rerun_helper.update_inertial_frame(axis_scale=2)

def run_visual_navigation():
    cap = cv2.VideoCapture("data/outdoor/calibration.MOV")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {width}x{height}, Number of frames: {num_frames}")
    exit()

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

        if use_rerun:

            # Convert GPS coordinates to ECEF
            ned = navpy.lla2ned(lat, lon, alt, lat_ref, lon_ref, alt_ref)
            print(f"North: {ned[0]}, East: {ned[1]}, Down: {ned[2]}")

            frame = draw_horizon(frame, width, height, system, CAMERA)
            # cv2.imshow("Good Points", frame)
            # cv2.waitKey(2000)
            
            # rr.set_time(timeline="Timeline", duration=t)

            # trajectory_enu.append(np.array([ned[1], ned[0], ned[2]]))

            # Log the camera extrinsics as a transform, this will move the camera frustrum to the correct position in the world
            # rr.log("world/camera_frame", rr.Transform3D(translation=[ned[0], ned[1], ned[2]],  rotation=rr.Quaternion(xyzw=(0, 0, 0, 1))))

            # Log a visible point at the camera origin
            # rr.log("world/camera_frame/origin", rr.Points3D([ned[0], ned[1], ned[2]], colors=np.array([[255, 0, 0]]), radii=0.05))
            # rr.log("world/gps_trajectory", rr.LineStrips3D([np.array(trajectory_enu)], colors=[[0, 255, 0]]))
            
            rerun_helper.update_body_frame(system=system, axis_scale=2)
            rerun_helper.update_gps_position(ned=ned)
            rerun_helper.update_camera_frustum(width, height, frame, CAMERA=CAMERA)
            frame = rerun_helper.draw_horizon(euler_angles=np.array(system.density.mean[9:12]), image=frame, image_height=height, image_width=width, CAMERA=CAMERA)
            # cv2.imshow("Good Points", frame)
            # cv2.waitKey(2000)

            rr.log("world/camera/image", rr.Image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
            
            # estimated_trajectory_enu.append(np.array([system.density.mean[6], system.density.mean[7], -system.density.mean[8]]))

            # Add state estimate to the rerun timeline.
            # rr.log("world/estimated_trajectory", rr.LineStrips3D([np.array(estimated_trajectory_enu)], colors=[[0, 0, 255]]))

            # rr.log("world/camera_frame/pinhole", rr.Pinhole(image_from_camera=CAMERA.matrix, resolution=(width, height), camera_xyz=rr.ViewCoordinates.RDF))
        
            # _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            # rr.log("/image/frame", rr.EncodedImage())
            # time.sleep(10)

        # 4. Update the previous frame and previous flow measurement for the next iteration.
        previous_frame = frame.copy()
        rQOikm1 = measurement_flow_bundle.data.rQOik.copy()
        time.sleep(100)
        
        # rr.log("world/chessboard_corners", rr.Points3D(grid_points, colors=np.array([[0, 255, 0]]), radii=0.0015))
        
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


def draw_horizon(frame, frame_width, frame_height, system, camera):
    """
    Draw the horizon on the image frame.
    """
    state = system.density.mean
    rpy = state[9:12]

    print(rpy)
    horizon_pts = []
    camera_horizontal_vectors = []
    num_samples = 60
    R = Rotations.rpy2rot(rpy, order='zxy').R
    K = CAMERA.matrix
    # Note the state vector roll pitch yaw states encode the rotation matrix Rnb, that is the rotation from world basis vectors into body basis vectors / frame 
    # Rab is the matrix that rotates the vectors of the basis a into the vectors of the basis b
    # Rnb = Rotations.rpy2rot(rpy).R          # {n} -> {b}    (world to body)
    # Rbc = camera.rotation_matrix.R          # {b} -> {c}    (body to camera)
    # Rnc = Rnb @ Rbc                         # {n} -> {c}    (world to camera)
    # Rnc = Rbc @ Rnb
    # Create direction vectors pointing out of the camera optical axis
    for i in range(num_samples):

        angle = 2 * np.pi * i / num_samples
        dir_c = [np.sin(angle), 0, np.cos(angle)]   # Camera XZ plane
        
        dir_c = R @ np.array(dir_c)
    
    if dir_c[2] > 0.02:  
        camera_horizontal_vectors.append(dir_c)
        p = K @ dir_c

        if p[2] > 0:
            p_img = p[:2] / p[2]
            u = int(round(p_img[0]))
            v = int(round(p_img[1]))
            if 0 <= u < frame_width and 0 <= v < frame_height:
                horizon_pts.append((u, v))

    # Draw on image
    if len(horizon_pts) > 5:
        pts = np.array(horizon_pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=False, 
                     color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    # rr.log(
    #     "world/inertial_frame/body_frame/camera_frame/horizon_from_camera",
    #     rr.Arrows3D(
    #         vectors=camera_horizontal_vectors,
    #         colors=[255, 100, 0],
    #         radii=0.02,
    #     )
    # )

    return frame

run_visual_navigation()
