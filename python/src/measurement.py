from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np
from src.rotations import Rotations
from src.gaussian import Gaussian
from src.measurement_flow_bundle import CAMERA
from numpy.typing import NDArray

from src.sensor_type import SensorType
from src.measurement_flow_bundle import MeasurementFlowBundle
e3 = np.array([0.0, 0.0, 1.0])

@dataclass
class MeasurementBase(ABC):
    """
    TODO:
    """

    @abstractmethod
    def as_vector(self) -> NDArray[np.float64]: ...


@dataclass
class MeasurementGPS(MeasurementBase):
    """
    TODO:
    """

    north: float
    east: float
    down: float
    velocity_north: float
    velocity_east: float
    velocity_down: float
    noise_density: np.ndarray = field(default_factory=lambda: np.diag([1, 1, 1, 0.1, 0.1, 0.1]))

    def as_vector(self):
        return np.array([self.north, self.east, self.down, self.velocity_north, self.velocity_east, self.velocity_down])


@dataclass
class MeasurementBaro(MeasurementBase):
    """
    TODO:
    """
    alt: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.array([[15.0]]))

    def log_likelihood(self, x):
        """
        TODO:
        """
        likelihood = self.predict_density(x)
        return likelihood.log_pdf(self.alt)

    def predict_density(self, x: np.ndarray):
        """
        TODO: 
        """
        alt = self.predict(x) # TODO: See if this should just be the measurement model from measurmeentmodel class
        return Gaussian(mean=alt, covariance=self.R)
    
    def predict(self, x: np.ndarray):
        """
        TODO:
        """
        # rBNnk = x[6:9]
        # Rnbk = Rotations.rpy2rot(x[9:12])
        # rCBb = CAMERA.translation_vector 
        # rCNnk = rBNnk + Rnbk.R @ rCBb
        # print("-x[8]:", -x[8])
        # print("np.array([-e3 @ rCNnk])", np.array([-e3 @ rCNnk]))
        # return np.array([-e3 @ rCNnk])
        return np.array([-x[8]])
    


    def as_vector(self):
        """TODO:
        Currently it is already a vector so just return the vector. 
        """
        return self.alt


@dataclass
class MeasurementMag(MeasurementBase):
    """
    TODO:
    """

    magx_mag: float
    noise_density: float = 0.0

    def as_vector(self):
        """TODO"""
        return np.array([])


@dataclass
class MeasurementGyro(MeasurementBase):
    """
    TODO:
    """

    gyro_x: float
    gyro_y: float
    gyro_z: float
    noise_density: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01, 0.01]))

    def as_vector(self):
        """TODO"""
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z])


@dataclass
class MeasurementAcc(MeasurementBase):
    """
    TODO:
    """

    specific_force_x: float
    specific_force_y: float
    specific_force_z: float
    noise_density: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1, 0.1]))

    def as_vector(self):
        """TODO"""
        return np.array([self.specific_force_x, self.specific_force_y, self.specific_force_z])

MeasurementData = Union[MeasurementGPS, MeasurementBaro, MeasurementMag, MeasurementGyro, MeasurementAcc, MeasurementFlowBundle]

@dataclass
class Measurement:
    """
    TODO:
    """

    time: float
    sensor: SensorType
    data: MeasurementData

    def as_vector(self) -> NDArray[np.float64]:
        """
        TODO:
        """
        return self.data.as_vector()
    
    
