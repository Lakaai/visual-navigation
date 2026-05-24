import cv2
import numpy as np
from src.camera import CAMERA

import cv2
import numpy as np

from src.gaussian import Gaussian
from src.gaussian_vector import GaussianVector
from src.rotations import Rotations, RotationMatrix
# from scipy.spatial.transform import Rotation

DIVISOR = 2
MAX_NUM_FEATURES = 1500
MIN_NUM_FEATURES = 1100
QUALITY_LEVEL = 0.0001
MIN_DISTANCE_PIXELS = 15

e3 = np.array([0.0, 0.0, 1.0])

class MeasurementFlowBundle:
    """
    TODO: 
    """
    def __init__(self, time: float, imgk_raw: np.ndarray, imgkm1_raw: np.ndarray, rQOikm1: np.ndarray):
        self.time = time
        self.rQOikm1 = rQOikm1
        self.rQOik = None
        self.pkm1 = None # Homogeneous coordinates of features in previous frame that satisfy the epipolar constraint
        self.pk = None # Homogeneous coordinates of features in current frame that satisfy the epipolar constraint
        self.imgk_raw = imgk_raw
        self.sigma = 2.25 # Standard deviation in pixel units

        imgk_gray = cv2.cvtColor(imgk_raw, cv2.COLOR_BGR2GRAY)
       
        # First frame, initialise features
        if rQOikm1 is None:
            print("First frame, initialising features!")

            rQOik = cv2.goodFeaturesToTrack(imgk_gray, maxCorners=MAX_NUM_FEATURES, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE_PIXELS)

            rQOik = cv2.cornerSubPix(imgk_gray, rQOik, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))

            self.rQOik = rQOik.astype(np.float32)

            # Construct the previous bundle of features for the next frame 
            self.rQOikm1 = self.rQOik.copy()
            return
        
        # Subsequent frames, track features
        else:
            imgkm1_gray = cv2.cvtColor(imgkm1_raw, cv2.COLOR_BGR2GRAY)

            # TODO: scale previous features for tracking this is what cpp does

            rQOik, status_vector, err = cv2.calcOpticalFlowPyrLK(
                imgkm1_gray,
                imgk_gray,
                self.rQOikm1,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            # Extract features that have been successfully tracked between both frames
            self.rQOik = rQOik[status_vector.ravel() == 1] # Current good points
            rQOikm1 = self.rQOikm1[status_vector.ravel() == 1] # Previous good points

            self.mask = np.zeros_like(imgk_raw)

            # If too many features are lost, detect new ones
            if self.rQOikm1.shape[0] < MIN_NUM_FEATURES:
                self.rQOik = self.detect_new_features(imgk_gray)

            # Calculate undistorted feature locations
            rQbarOik = cv2.undistortPoints(self.rQOik, CAMERA.matrix, CAMERA.distortion_coeffs, P=CAMERA.matrix)
            rQbarOikm1 = cv2.undistortPoints(rQOikm1, CAMERA.matrix, CAMERA.distortion_coeffs, P=CAMERA.matrix)

            # Use RANSAC to find fundamental matrix and determine inliers
            self.F, mask = cv2.findFundamentalMat(rQbarOikm1, rQbarOik, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

            # The mask returned by findFundamentalMat has 1 for inliers and 0 for outliers, we want to keep only inliers
            # print("Number of features before RANSAC rQbarOk: ", rQbarOik.shape[0])
            num_inliers = np.sum(mask)

            mask = mask.ravel().astype(bool)

            # Reshape OpenCV format (N,1,2) → (2,N)
            rQbarOik   = rQbarOik.reshape(-1, 2).T
            rQbarOikm1 = rQbarOikm1.reshape(-1, 2).T

            # print("rQbarOik shape:", rQbarOik.shape)  # should be (2, N)

            # Select COLUMNS (this is the fix) TODO: Create an example script to understand this
            inliers_k   = rQbarOik[:, mask]        # (2, n_inliers)
            inliers_km1 = rQbarOikm1[:, mask]      # (2, n_inliers)

            # Convert to homogeneous
            self.pk   = np.vstack([inliers_k,   np.ones((1, num_inliers))])
            self.pkm1 = np.vstack([inliers_km1, np.ones((1, num_inliers))])
        return
    
    def detect_new_features(self, imgk_gray):
        print("Not enough features: ",  self.rQOikm1.shape[0])
        rQOik = cv2.goodFeaturesToTrack(imgk_gray, maxCorners=MAX_NUM_FEATURES, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE_PIXELS)
        rQOik = cv2.cornerSubPix(imgk_gray, rQOik, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))

        print("Adding features: ", self.rQOik.shape[0] - self.rQOikm1.shape[0])
        # TODO: Investigate TRYING TO GRAB NEW FEATURES CAN ACTUALLYNG END UP FINDING LESS FEATURES
        return rQOik.astype(np.float32)
        

    def plot_inliers(self, imgk_raw, mask):
        """
        TODO:
        """
        # img = cv2.add(self.mask, self.imgk_raw)
        for i in range(self.pk.shape[0]):
            x, y = self.pk[i].ravel()
            cv2.circle(imgk_raw, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Plot outliers in red
        for i in range(self.rQOik.shape[0]):
            if mask[i] == 0:
                x, y = self.rQOik[i].ravel()
                cv2.circle(imgk_raw, (int(x), int(y)), 3, (0, 0, 255), -1)

        cv2.imshow("Inlier Features", imgk_raw)
        cv2.waitKey(33)

    def log_likelihood(self, x: np.ndarray) -> float:
        """
        TODO: The measurement flow bundle log likelihood, returns p(y | x).
        """
        total_log_likelihood = 0.0

        # Convert homogeneous to Euclidean (N, 2)
        pk = self.pk.T
        rQbarOk = pk[:, :2] / pk[:, 2:3] 

        # Get the measurement likelihood p(y | x) 
        likelihood = self.predict_density(x)

        # Evaluate how probable the observed measurement is under the predicted distribution
        total_log_likelihood = np.sum(likelihood.log_pdf(rQbarOk)) 

        # rQbarOk_hat = likelihood.mean
        # self.print_pixel_error(rQbarOk, rQbarOk_hat)
        # self.plot_predicted_measurements(rQbarOk, rQbarOk_hat)
        return total_log_likelihood
    
    
    def predict(self, x: np.ndarray):
        """
        TODO:
        Uses homogenous point homogrophies for the ground and sky dome
        """
        assert len(x) == 18

        rBNnk = x[6:9]
        rBNnkm1 = x[12:15]

        Rnbk = Rotations.rpy2rot(x[9:12])
        Rnbkm1 = Rotations.rpy2rot(x[15:18])

        rCBb = CAMERA.translation_vector 
        Rbc = CAMERA.rotation_matrix

        rCNnk = rBNnk + Rnbk.R @ rCBb
        rCNnkm1 = rBNnkm1 + Rnbkm1.R @ rCBb
        Rnck = Rnbk.R @ Rbc.R
        Rnckm1 = Rnbkm1.R @ Rbc.R

        K = CAMERA.matrix

        delta = rCNnkm1 - rCNnk

        Hz = K @ Rnck.T @ (np.eye(3) - np.outer(delta, e3) / rCNnkm1[2]) @ Rnckm1 @ np.linalg.inv(K)
        Hinf = K @ Rnck.T @ Rnckm1 @ np.linalg.inv(K)

        # Compute projected Z value for each point
        z_projs = (Rnck @ np.linalg.inv(K) @ self.pk)[2, :]

        # Compute both potential sets of results
        pk_hat_below = Hz @ self.pkm1  
        pk_hat_above = Hinf @ self.pkm1

        mask = z_projs > 0

        # Select Hj(xk) based on the mask (equation 6b)
        pk_hat = np.where(mask, pk_hat_below, pk_hat_above)

        return pk_hat.T


        # INITIAL IMPLEMENTATION 
        # z_proj = (Rnck @ np.linalg.inv(K) @ self.pk)[2]
        
        # if z_proj > 0:
        #     # Point is below horizon, use zero homography
        #     Hz = K @ Rnck.T @  (np.eye(3) - np.outer(delta, e3) / rCNnkm1[2]) @ Rnckm1 @ np.linalg.inv(K)
        #     pk_hat = Hz @ self.pkm1

        # else:
        #     # Point is above horizon, use infinity homography
        #     Hinf = K @ Rnck.T @ Rnckm1 @ np.linalg.inv(K)
        #     pk_hat = Hinf @ self.pkm1
        
        # return pk_hat


    def predict_density(self, x: np.ndarray):
        """
        TODO: 

        :param x: TODO:
        :type x: np.ndarray
        :param pki: ith point in the measurement flow bundle for the current frame
        :type pki: np.ndarray 
        :param pkm1i: ith point in the measurement flow bundle for the previous frame
        :type pkm1i: np.ndarray

        """
        # Get prediction in homogeneous coordinates (N , 3)
        pk_hat = self.predict(x)
        
        # Convert homogeneous to Euclidean (N, 2)
        rQbarOik_hat = pk_hat[:, :2] / pk_hat[:, 2:3] 

        N = len(rQbarOik_hat)

        # Create a stack of covariance matrices (N, 2, 2)
        covariances = np.eye(2).reshape(1, 2, 2) * (self.sigma * self.sigma)
        covariances = np.repeat(covariances, N, axis=0)

        return GaussianVector(mean=rQbarOik_hat, covariance=covariances)

    def as_vector(self):
        """
        TODO:
        """
        return self.pk
    
 
    # # Create some random colors
    # color = np.random.randint(0, 255, (100, 3))

    # # Take first frame and find corners in it
    # ret, old_frame = cap.read()
    # old_gray = cv2.cvtColor(old_frame,
    #                         cv2.COLOR_BGR2GRAY)

    # # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)

    # while(1):

    #     ret, frame = cap.read()
    #     frame_gray = cv2.cvtColor(frame,
    #                               cv2.COLOR_BGR2GRAY)

    #     # calculate optical flow
    #     next_points, status_vector, err = cv2.calcOpticalFlowPyrLK(old_gray,
    #                                            frame_gray,
    #                                            p0, None,
    #                                            **lk_params)

    #     # Select good points
    #     good_new = next_points[status_vector == 1]
    #     good_old = p0[status_vector == 1]

    #     # draw the tracks
    #     for i, (new, old) in enumerate(zip(good_new,
    #                                        good_old)):
    #         a, b = new.ravel()
    #         c, d = old.ravel()
    #         mask = cv2.line(mask, (a, b), (c, d),
    #                         color[i].tolist(), 2)

    #         frame = cv2.circle(frame, (a, b), 5,
    #                            color[i].tolist(), -1)

    #     img = cv2.add(frame, mask)

    #     cv2.imshow('frame', img)

    #     k = cv2.waitKey(25)
    #     if k == 27:
    #         break

    #     # Updating Previous frame and points
    #     old_gray = frame_gray.copy()
    #     p0 = good_new.reshape(-1, 1, 2)

    # def log_likelihood(self, grad: bool = False):
    #     """

    #     Computes the log-likelihood `log p(y | x)` of a measurement under the predicted measurement model,
    #     optionally returning the gradient with respect to state `x`.

    #     TODO:
    #     # Arguments
    #     - `x::AbstractVector`: The current state estimate.
    #     - `measurement::AbstractVector`: The actual observed measurement.
    #     - `grad::Bool`: If `true`, also returns the gradient ∂/∂x log p(y | x).

    #     # Returns
    #     - If `grad=false`: Returns `log_likelihood::Float64`.
    #     - If `grad=true`: Returns a tuple `(log_likelihood, gradient)` where:
    #         - `gradient::Vector`: Gradient of the log-likelihood with respect to state.

    #     # Notes
    #     - Assumes Gaussian measurement noise.
    #     - Uses the chain rule: ∂/∂x log p(y | x) = -Jᵀ ∂/∂y log N(y; h(x), R)
    #     - Evaluates log N(y; h(x), R) and d/dy log N(y; h(x), R)
    #     """
    #     # likelihood, dhdx = self.predict_density(x)      # Compute the measurement likelihood p(x∣z) = N(y; h(x), R)

    #     # loglikelihood = Gaussian.log_pdf(self, )

    #     n = len(self.rQOik)
    #     rQbarOik = self.rQOik

    #     # Process each set of points in the measurement vector independently
    #     for i in range(n):
    #         # print(f"Processing point {i} of {n}")
    #         # print("rQbarOik: ", rQbarOik[i])

    #         # print(type(rQbarOik[i]))
    #         # print(rQbarOik[i].shape)

    #         # Single point measurement density
    #         likelihood = predict_density(x, system, i)
    #         likelihood = Gaussian(np.array([0]), np.array([0]))

    #         # total_log_likelihood += Gaussian.log_pdf(rQbarOik, likelihood)

    #     exit()
    #     return total_log_likehood

    def plot_predicted_measurements(self, rQbarOk, rQbarOk_hat):
        """
        Plot the predicted and observed measurements

        :param rQbarOk: The observed measurements as (N, 2) 
        :type rQbarOk: np.ndarray TODO 
        :param rQbarOk_hat: The predicted measurements as (N, 2)
        :type rQbarOk_hat: np.ndarray TODO
        """ 
        for i in range(rQbarOk.shape[0]):
            
            x_obs, y_obs = rQbarOk[i, :]
            pt_obs = (int(x_obs), int(y_obs))
            cv2.circle(self.imgk_raw, pt_obs, 3, (0, 255, 0), -1)

            x_pred, y_pred = rQbarOk_hat[i, :]
            pt_pred = (int(x_pred), int(y_pred))
            cv2.circle(self.imgk_raw, pt_pred, 3, (0, 0, 255), -1)

        cv2.imshow("Observed (green) vs Predicted (red)", self.imgk_raw)
        cv2.waitKey(33)

    def print_pixel_error(self, rQbarOk, rQbarOk_hat):
        errors = rQbarOk - rQbarOk_hat 
        distances = np.linalg.norm(errors, axis=0) # (N,)
        mean_pixel_error = np.mean(distances)
        print(f"Mean Prediction Error: {mean_pixel_error:.4f} pixels")
        
