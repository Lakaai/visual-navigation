from enum import Enum


class SensorType(Enum):
    IMU = "IMU"
    GPS = "GPS"
    BARO = "Baro"
    MAG = "Mag"
    GYRO = "Gyro"
    ACC = "Acc"
    CAMERA = "Camera"