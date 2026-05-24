"""
image coordinates (u, v) 
The (0,0) coordinate is the top-left corner

"""
from dataclasses import dataclass
import numpy as np
from src.rotations import Rotations, RotationMatrix
import cv2 

@dataclass
class Camera():
    """"
    
    """
    matrix: np.ndarray
    distortion_coeffs: np.ndarray
    translation_vector: np.ndarray
    rotation_matrix: RotationMatrix


    def undistort_features(self, features):
        """
        TODO: 
        """
        undistorted_features = cv2.undistortPoints(features, self.matrix, self.distortion_coeffs, P=self.matrix)
        return undistorted_features
    
    def is_vector_within_fov(self, rPCc: np.ndarray) -> bool:
        """
        rPCc : position of point in camera frame
        Assumes camera is looking along its z-axis and that the image plane is at z=1 in the camera frame

        Note if two vectors have unit length the dot product cos(theta) = A dot B/(norm(A)*norm(B)) simplifies toA dot B 
        so we can compute the angle between two vectors by normalising them and then taking the arccos of their dot product. 
        """
        uPCc = rPCc / np.linalg.norm(rPCc)

        principal_direction = np.array([0, 0, 1])

        # Compute horizontal angle
        horizontal_projection = np.array([uPCc[0], 0, uPCc[2]])
        horizontal_projection = horizontal_projection / np.linalg.norm(horizontal_projection)
        horizontal_angle = np.arccos(np.dot(horizontal_projection, principal_direction))

        # Compute vertical angle
        vertical_projection = np.array([0, uPCc[1], uPCc[2]])
        vertical_projection = vertical_projection / np.linalg.norm(vertical_projection)
        vertical_angle = np.arccos(np.dot(vertical_projection, principal_direction))

        # Convert angles to degrees
        horizontal_angle_deg = horizontal_angle * 180.0 / np.pi
        vertical_angle_deg = vertical_angle * 180.0 / np.pi

        # Project the vector onto the image plane
        pixel = self.vector_to_pixel(uPCc)

        # Check if the projected point is within the image bounds
        inside_image = (pixel[0] >= 0 and pixel[0] < self.matrix[0, 2] * 2 and
                        pixel[1] >= 0 and pixel[1] < self.matrix[1, 2] * 2)

        # Check if both angles are within their respective FOV limits
        inside_horizontal_fov = horizontal_angle_deg <= self.hFOV / 2.0
        inside_vertical_fov = vertical_angle_deg <= self.vFOV / 2.0

        return inside_image and inside_horizontal_fov and inside_vertical_fov
    
    def vector_to_pixel(self, rPCc):
        """
        TODO: 
        Compute the pixel location (rQOi) for the given unit vector (uPCc).

        Reference: https://www.geeksforgeeks.org/computer-vision/mapping-coordinates-from-3d-to-2d-using-opencv-python/
        """
        uPCc = rPCc / np.linalg.norm(rPCc)

        # IMPORTANT: Correct shape for cv2.projectPoints -> (1, 1, 3)
        object_points = np.array([[uPCc[0], uPCc[1], uPCc[2]]], dtype=np.float32).reshape(1, 1, 3)
        print(f"Object points (camera coordinates): {object_points}")
        print("Object shape:", object_points.shape)
        # Since point is already in camera coordinates, rvec and tvec = zero
        rvec = np.zeros((3, 1), dtype=np.float32)   # No rotation
        tvec = np.zeros((3, 1), dtype=np.float32)   # No translation

        image_points, _ = cv2.projectPoints(
            object_points, 
            rvec, 
            tvec, 
            self.matrix, 
            self.distortion_coeffs
        )

        # image_points shape is (1, 1, 2) → extract the pixel
        pixel = image_points[0][0]   # Returns [u, v]
        return pixel

    def pixel_to_vector(self, rQOi, camera): 
        """ TODO: 
        Compute unit vector (uPCc) for the given pixel location (rQOi).
        """

        image_points = np.array([[rQOi[0], rQOi[1]]], dtype=np.float32)

        # Undistort and normalise the point
        normalised_points = cv2.undistortPoints(image_points, camera.camera_matrix, camera.dist_coeffs)

        # The normalised point is now [x, y, 1] in camera coordinates
        uPCc = np.array([normalised_points[0][0][0], normalised_points[0][0][1], 1.0])

        # Normalise to unit vector
        return uPCc / np.linalg.norm(uPCc)
        


CAMERA = Camera(
matrix=np.array([
    [2230.0302561729463, 0.0, 1326.9024252144795],
    [0.0, 2249.3015295213522, 483.88958360970497],
    [0.0, 0.0, 1.0]
    ]), 
    distortion_coeffs=np.array([
        0.010524588721904289, 0.048537210858010528, -0.059423525226001202,
        -0.0063860373221385318, 0.11992593543150673, -0.016224313722952306,
        0.1179079716819904, 0.06468044932840758, 0.015143083729458221,
        -0.011402957916733165, 0.052841065220381671, 0.0074396821766865989
    ]), 
    translation_vector=np.array([0.0, 0.0, 0.0]),
    rotation_matrix=RotationMatrix(np.array([
            # [0, 1, 0],   # c1 = b2
            # [0, 0, 1],   # c2 = b3
            # [1, 0, 0]    # c3 = b1
            [0, 0, 1],  # b1 = c3
            [1, 0, 0],  # b2 = c1
            [0, 1, 0],  # b3 = c2
        ]))
    )


matrix = np.array([
    [2230.0302561729463, 0.0, 1326.9024252144795],
    [0.0, 2249.3015295213522, 483.88958360970497],
    [0.0, 0.0, 1.0]
], dtype=np.float32)   # ← Force float32 here

distortion_coeffs = np.array([
    0.010524588721904289, 0.048537210858010528, -0.059423525226001202,
    -0.0063860373221385318, 0.11992593543150673, -0.016224313722952306,
    0.1179079716819904, 0.06468044932840758, 0.015143083729458221,
    -0.011402957916733165, 0.052841065220381671, 0.0074396821766865989
], dtype=np.float32)   # ← Force float32 here

translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)

rotation_matrix = RotationMatrix(np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=np.float32))

# Create camera with proper types
camera = Camera(
    matrix=matrix,
    distortion_coeffs=distortion_coeffs,
    translation_vector=translation_vector,
    rotation_matrix=rotation_matrix
)

vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
pixel = camera.vector_to_pixel(vec)
print("Projected center pixel:", pixel)

cap = cv2.VideoCapture("data/outdoor/flight.MOV")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video resolution: {width}x{height}, Number of frames: {num_frames}")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


# Read frame 
ret, frame = cap.read()

print("Frame shape:", frame.shape)
# Draw the projected point on the frame
if frame is not None:
    # Convert pixel to integer coordinates
    u, v = int(pixel[0]), int(pixel[1])
    
    # Draw a circle at the projected point
    cv2.circle(frame, (u, v), radius=8, color=(0, 255, 0), thickness=-1)   # filled green circle
    cv2.circle(frame, (u, v), radius=12, color=(0, 0, 255), thickness=2)   # red outline
    
    # Optional: Draw crosshair
    cv2.line(frame, (u-20, v), (u+20, v), (0, 0, 255), 2)
    cv2.line(frame, (u, v-20), (u, v+20), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Projected Point on Frame", frame)
    cv2.waitKey(0)          # Press any key to close
    cv2.destroyAllWindows()

