""" TODO """

import rerun as rr
import numpy as np
import cv2
from src.system_estimator import SystemEstimator
from scipy.spatial.transform import Rotation
class RerunHelper:
    """
    TODO:  
    """
    def __init__(self):
        self.true_previous_positions = []      # Manual history of previous positions
        self.true_rBNkm1 = None

        self.previous_positions = []           # History of previous positions from the state estimate
        self.rBNkm1 = None
        
    
    def update_inertial_frame(self, axis_scale=2):
        """ 
        TODO: 
        """
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

    def update_gps_position(self, ned, axis_scale=2):
        """
        TODO: 
        """

        #  TODO: Extract previous body position from the state etakm1 
        self.true_previous_positions.append(ned)
            
        if len(self.true_previous_positions) > 150:
            self.true_previous_positions = self.true_previous_positions[-150:]

        if len(self.true_previous_positions) > 0:
            rr.log("world/inertial_frame/previous_gps_position", rr.Points3D(np.array(self.true_previous_positions), colors=np.array([[0, 0, 255]]), radii=0.05))

    def update_body_frame(self, system: SystemEstimator, axis_scale=2):
        """
        TODO: Estimated body from visualiser 
        """

        #  TODO: Extract previous body position from the state etakm1 
        if self.rBNkm1 is not None:
            self.previous_positions.append(self.rBNkm1.copy())
            
            # Limit history length
            if len(self.previous_positions) > 150:
                self.previous_positions = self.previous_positions[-150:]

        if len(self.previous_positions) > 0:
            rr.log("world/inertial_frame/previous_body_position", rr.Points3D(np.array(self.previous_positions), colors=np.array([[255, 0, 0]]), radii=0.05))

        Rnb = Rotation.from_euler('xyz', system.density.mean[9:12])
        rBNn=system.density.mean[6:9]
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
        self.rBNkm1 = rBNn

    def update_camera_frustum(self, width, height, image, CAMERA):
        """
        rCBn : position of camera in navigation frame (NED)
        Rbc  : Body → Camera rotation
        Note when using rr.Transform3D the translations are relative to the parent's origin, for example the parent for the camera_frame is the body_frame therefore
        the translation is rCBb
        """
        Rbc = Rotation.from_matrix(CAMERA.rotation_matrix.R)
        rCBb = CAMERA.translation_vector
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

        
    def draw_horizon(self, euler_angles, image, image_height, image_width, CAMERA):
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