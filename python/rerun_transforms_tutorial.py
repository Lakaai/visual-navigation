"""
TODO:

The script is a tutorial on how to visualise transforms (rotations and translations) using the rerun.
"""

import rerun as rr
import rerun.blueprint as rrb
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from src.camera import CAMERA

def draw_horizon(euler_angles, image, image_height, image_width):
    """
    Rnb: world -> body
    Rbc: body -> camera
    rBNn: body position in world
    rCBb: camera position in body frame

    Note the state vector roll pitch yaw states encode the rotation matrix Rnb, that is the rotation from world basis vectors into body basis vectors / frame 
    Rab is the matrix that rotates the vectors of the basis a into the vectors of the basis b
    """                     
    K = CAMERA.matrix
    
    horizon_pts = []
    camera_horizontal_vectors = []
    num_samples = 60

    R = Rotation.from_euler("zxy", euler_angles, degrees=True)

    # Create direction vectors pointing out of the camera optical axis
    for i in range(num_samples):
        angle = 2 * np.pi * i / num_samples
        dir_c = [np.sin(angle), 0, np.cos(angle)]   # Camera XZ plane
       
        dir_c = R.apply(dir_c)
        
        

        if dir_c[2] > 0.02:  
            camera_horizontal_vectors.append(dir_c)
            p = K @ dir_c

            if p[2] > 0:
                p_img = p[:2] / p[2]
                u = int(round(p_img[0]))
                v = int(round(p_img[1]))
                if 0 <= u < image_width and 0 <= v < image_height:
                    horizon_pts.append((u, v))

    # Draw on image
    if len(horizon_pts) > 5:
        pts = np.array(horizon_pts, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=False, 
                     color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    rr.log(
        "world/inertial_frame/body_frame/camera_frame/horizon_from_camera",
        rr.Arrows3D(
            vectors=camera_horizontal_vectors,
            colors=[255, 100, 0],
            radii=0.02,
        )
    )

    return image

def plot_inertial_frame(axis_scale=2):
    """ """
    rr.log(
        "world/inertial_frame/axes",
        rr.Arrows3D(
            vectors=[[axis_scale, 0, 0], [0, axis_scale, 0], [0, 0, axis_scale]],
            colors=[
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
        ),
    )

    rr.log(
        "world/inertial_frame/axis_labels",
        rr.Points3D(
            positions=[
                [axis_scale, 0, 0],
                [0, axis_scale, 0],
                [0, 0, axis_scale],
                [0, 0, 0],
            ],
            labels=["X", "Y", "Z", "Inertial"],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        ),
    )


def plot_body_frame(rBNn: np.ndarray, Rnb: np.ndarray, axis_scale=2):
    """ """

    rr.log(
        "world/inertial_frame/body_frame",
        rr.Transform3D(
            translation=rBNn,
            rotation=rr.Quaternion(xyzw=Rnb.as_quat()),
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/axes",
        rr.Arrows3D(
            vectors=[[axis_scale, 0, 0], [0, axis_scale, 0], [0, 0, axis_scale]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/axis_labels",
        rr.Points3D(
            positions=[
                [axis_scale, 0, 0],
                [0, axis_scale, 0],
                [0, 0, axis_scale],
                [0, 0, 0],
            ],
            labels=["X", "Y", "Z", "Body"],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        ),
    )


def plot_camera_frustum(width, height, image, rCBb, Rbc):
    """
    rCBn : position of camera in navigation frame (NED)
    Rbc  : Body → Camera rotation
    Note when using rr.Transform3D the translations are relative to the parent's origin, for example the parent for the camera_frame is the body_frame therefore
    the translation is rCBb
    """

    rr.log(
        "world/inertial_frame/body_frame/camera_frame",
        rr.Transform3D(
            translation=rCBb,
            rotation=rr.Quaternion(xyzw=Rbc.as_quat()),
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/camera_frame/pinhole",
        rr.Pinhole(
            resolution=[width, height],
            image_from_camera=CAMERA.matrix,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.8,
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/camera_frame/pinhole/image",
        rr.Image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
    )


def visualise_transforms():
    """ 
    # The columns of a rotation matrix are the subscript basis vectors expressed in the superscript coordinates
    """

    # Define the body roll pitch yaw euler angles
    rpy = np.array([0.5, -5, 0])

    # Intialise world frame as NED (Right-Handed Z-Down)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)

    # Define the body to camera rotation matrix
    Rbc = Rotation.from_matrix(
        np.array(
            [
                [0, 0, 1],  # b1 = c3
                [1, 0, 0],  # b2 = c1
                [0, 1, 0],  # b3 = c2
            ]
        )
    )

    # Define navigation to body rotation matrix
    # Rnb = Rotation.from_euler("xyz", rpy, degrees=True)
    # print("Rnb:\n", Rnb.as_matrix())
    Rnb = Rotation.from_euler("zyx", rpy, degrees=True)
    print("Rnb:\n", Rnb.as_matrix())
    rBNn = np.array([3, 3, -3])
    rCBb = np.array([0.1, 0.1, -0.1])   # Apply some offset only for visualisation purposes

    cap = cv2.VideoCapture("data/outdoor/flight.MOV")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    ret, frame = cap.read()

    plot_inertial_frame()

    plot_body_frame(rBNn=rBNn, Rnb=Rnb, axis_scale=2)

    annotated_image = draw_horizon(euler_angles=rpy, image=frame, image_height=height, image_width=width)

    rr.log("world/camera/image", rr.Image(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)))

    plot_camera_frustum(width=width, height=height, rCBb=rCBb, Rbc=Rbc, image=annotated_image)


def main():

    parser = argparse.ArgumentParser(description="TODO")
    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="World"),
        rrb.Spatial2DView(origin="world/camera", name="Camera"),
    )

    rr.script_setup(args, "transform_visualisation", default_blueprint=blueprint)

    visualise_transforms()

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
