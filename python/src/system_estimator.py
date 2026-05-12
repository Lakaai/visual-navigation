from enum import auto
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from src.update_method import UpdateMethod
from src.gaussian import Gaussian
from src.rotations import Rotations
from src.measurement import Measurement
from src.sensor_type import SensorType
from src.measurement_models import MeasurementModels
from src.measurement_flow_bundle import MeasurementFlowBundle

previous_time = 0


class SystemEstimator:
    def __init__(self, density: Gaussian):
        self.density = density
        self.previous_time = None
        self.Qeta = np.diag([2, 2, 2, 0.02, 0.02, 0.02])

    def dynamics(self, x):
        """
        TODO:
        The state vector x = [nu(6), eta(6)].T
        """
        nu = x[0:6]
        eta = x[6:12]

        Jk = self.construct_euler_rate_matrix(eta)

        nudot = np.zeros(6)
        etadot = Jk @ nu

        # TODO SHOULD ETAD ACTUALLY BE APPENDED TO THE END? CHECK THIS STATE VEC 
        return np.concatenate((nudot, etadot, eta))

    def construct_euler_rate_matrix(self, eta):
        """
        TODO:
        """
        thetanb = eta[3:6]
        # print("thetanb", thetanb)
        Rnb = Rotations.rpy2rot(thetanb)

        Jk = np.zeros((6, 6))

        phi = eta[3]
        theta = eta[4]

        Jk[0:3, 0:3] = Rnb.R

        Jk[3:6, 3:6] = [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]

        # print("Jk: ")
        # for row in Jk:
        #     print(f"| {' '.join(f'{val:6.2f}' for val in row)}   |")

        return Jk

    def rk4_step(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        TODO
        """
        # print("x inside rk4", x)
        k1 = self.dynamics(x)
        k2 = self.dynamics(x + k1 * dt / 2)
        k3 = self.dynamics(x + k2 * dt / 2)
        k4 = self.dynamics(x + k3 * dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    def predict(self, current_time: float, update_method: UpdateMethod):
        """
        TODO
        """
        if self.previous_time == None:
            print("Initialising previous time")
            self.previous_time = current_time

        dt = current_time - self.previous_time
        self.previous_time = current_time

        if dt <= 0:
            print("dt = 0, skipping prediction")
            return self.density

        def process_model(x):
            return self.rk4_step(x, dt)

        if update_method == UpdateMethod.UNSCENTED:
            # print("mean before prediction", self.density.mean)
            predicted_density = self.density.unscented_transform(process_model)

        elif update_method == UpdateMethod.AFFINE:
            # print("mean before prediction", self.density.mean)
            predicted_density = self.density.affine_transform(process_model)

            # TODO UNCOMMENT AND FIX 
            # Qd = np.zeros((12, 12))
            # Qd[6:12, 6:12] = self.Qeta * dt

            # # full_density_convariance = np.eye(18)
            # # full_density_covariance[0:12, 0:12] = predicted_density.covariance

            # print("mean before rebuild", self.density.mean)

            # # Rebuild the full 18 state density and covaraince.
            # self.density.mean[0:12] = predicted_density.mean
            # self.density.covariance[0:12, 0:12] = predicted_density.covariance + Qd  # Inject noise

            # print("convariance after prediction and rebuild")
            # for row in self.density.covariance:
            #     print(f"| {' '.join(f'{val:6.2f}' for val in row)}   |")

            # print("mean after prediction and rebuild", self.density.mean)
            self.density = predicted_density
            return

        else:
            raise ValueError(f"Invalid update method: {update_method}")

    def update(self, measurement: Measurement, update_method: UpdateMethod):
        """
        TODO:
        """
        print("Performing measurement update: ", measurement.sensor)

        if update_method == UpdateMethod.BFGS:
            
            # Minimise the negative log posterior to find the MAP estimate

            # TODO: Cost joint density may be better as a method of measurement class since it is not specific to the flow bundle measurment. 
            cost_function = lambda x: self.negative_log_posterior(x, self.density, measurement=measurement)

            initial_guess = self.density.mean
            if measurement.sensor == SensorType.CAMERA:
                if measurement.data.pk is not None:
                    # measurement.data.predict_density(self.density.mean)
                    result = minimize(fun=cost_function, x0=initial_guess, args=(), method='BFGS', options={'gtol': 1e-3, 'disp': True, 'maxiter':100})
                    # print(result)
                    cov = result.hess_inv
                    cov = 0.5 * (cov + cov.T)
                    self.density = Gaussian(result.x, cov)

            elif measurement.sensor == SensorType.BARO:
                result = minimize(fun=cost_function, x0=initial_guess, args=(), method='BFGS', options={'gtol': 1e-3, 'disp': True, 'maxiter':100})
                # print(result)
                cov = result.hess_inv
                cov = 0.5 * (cov + cov.T)
                self.density = Gaussian(result.x, cov)

            return 


        n = len(measurement.as_vector())
        idx_y = np.arange(0, n)
        idx_x = np.arange(n, n + 18)

        measurement_model = self.generate_measurement_model(measurement)

        # Create joint density maybe?
        augmented_predict_measurement = lambda x: self.augmented_predict_measurement(x, measurement_model)
        
        # Form the joint density p(xk, yk | y1:k-1) by propagating the prior p(xk | y1:k-1) through the transformation
        if update_method == UpdateMethod.UNSCENTED:
            pyx = self.density.unscented_transform(augmented_predict_measurement)

        elif update_method == UpdateMethod.AFFINE:
            pyx = self.density.affine_transform(augmented_predict_measurement)

        elif update_method == UpdateMethod.BFGSTRUSTSQRT:
            
            # Create cost function with prototype V = cost_function(x, g)
            # TODO: Cost joint density may be better as a method of measurement class since it is not specific to the flow bundle measurment. 
            cost_function = lambda x, g: MeasurementFlowBundle.cost_joint_density(x, self.density, g)
            return
        else:
            raise ValueError(f"Invalid update method: {update_method}")
        
        pyx.covariance[np.ix_(idx_y, idx_y)] += measurement.data.R

        # Symmetrise the covariance matrix to ensure it's positive definite
        pyx.covariance = 0.5 * (pyx.covariance + pyx.covariance.T)
        
        # Compute the conditional density p(xk | yk) by conditioning the joint density on the measurement yk
        # print(idx_x, idx_y)
        # self.print_matrix("covariance before cond:", self.density.covariance)
        # print("mean before cond:", pyx.mean)
        density = pyx.conditional(idx_x, idx_y, measurement.as_vector())
        # self.print_matrix("covariance after cond:", density.covariance)
        # print("mean after cond:", density.mean)
        # density.covariance = (density.covariance + density.covariance.T) / 2
        # print("shape of cov:", density.covariance.shape)
        # print("Is symmetric:" , np.allclose(density.covariance, density.covariance.T))
        # print("IS pos def:", np.linalg.cholesky(density.covariance)) # Check pos def
        
        self.density = density

        return
        

    
    def augmented_predict_measurement(self, x, measurement_model):
        """
        TODO:
        """

        y = measurement_model(x)

        h_augmented = np.concatenate((y, x))

        return h_augmented

 
    def negative_log_posterior(self, x: np.ndarray, density: Gaussian, measurement: Measurement) -> float:
        """
        TODO: 

        Defines the cost function for MAP estimator to minimise. The cost function is defined as the negative log posterior, that is the
        negative log of the measurement likelihood plus the negative log prior.

        V(xk) = -log p(yk | xk) - log p(xk; muk|k-1, Pk|k-1).

        For more information see slide 50 Gaussian Filtering I.

        :param x: The state at which to evaluate the negative log posterior, typically a hypothetical or candidate state estimate being searched over during optimisation.
        :type x: np.ndarray[float]
        :param distribution: TODO
        :type distribution: Gaussian
        :param measurement: TODO
        :type measurement: Measurement
        :param gradient: TODO
        :type gradient: bool
        :return: Negative log posterior
        :rtype: float
        """
        log_prior = density.log_pdf(x)
        log_likelihood = measurement.data.log_likelihood(x)
        return -(log_prior + log_likelihood)


    
    def generate_measurement_model(self, measurement: Measurement) -> Callable:
        """
        Generate the appropriate measurement model function based on the sensor type of the measurement.

        :param measurement: The measurement for which to generate the measurement model.
        :type measurement: Measurement
        :return: The measurement model function corresponding to the sensor type of the measurement.
        :rtype: Callable
        """
        print(measurement.sensor)
        measurement_model = {
            # SensorType.CAMERA: MeasurementFlowBundle.predict(measurement),
            SensorType.GPS: MeasurementModels.gps,
            SensorType.BARO: MeasurementModels.baro,
            # SensorType.BARO: measurement.data.predict_density,
            SensorType.MAG: MeasurementModels.mag,
            SensorType.ACC: MeasurementModels.accelerometer,
            SensorType.GYRO: MeasurementModels.gyroscope,
        }.get(measurement.sensor)

        if measurement_model is None:
            raise KeyError(f"Invalid sensor type: {measurement.sensor}")

        return measurement_model

    def print_matrix(self, string, matrix):
        print(f"{string}")
        for row in matrix:
            print(f"| {' '.join(f'{val:6.3f}' for val in row)}   |")
