import re
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pysrt

@dataclass 
class MeasurementDataSRT:
    timestamp: np.ndarray[float]
    altitude: np.ndarray[float]
    latitude: np.ndarray[float]
    longitude: np.ndarray[float]

def parse_srt(file_path: str):
    """
    TODO: This SRT parser assumes that the timesteps are in the format "2021-09-29 11:13:38,138,602" where the last two parts are milliseconds and microseconds respectively.
    """
    subtitles = pysrt.open(file_path)

    timestamp_seconds = []
    altitude = []
    latitude = []
    longitude = []

    for index, subtitle in enumerate(subtitles):
        
        timestamp_match = re.search(r"\d{2}:\d{2}:\d{2},\d+,\d+", subtitle.text)

        timestamp = None
        if timestamp_match:
            raw_time = timestamp_match.group()
            
            # Convert to standard format: replace last comma with dot
            # "2021-09-29 11:13:38,138,602" → "2021-09-29 11:13:38.138602"
            timestep_clean = raw_time.replace(",", ".", 1).replace(",", "")
            
            timestamp = datetime.strptime(timestep_clean, "%H:%M:%S.%f")

            # Convert to seconds since midnight
            timestamp_seconds.append(timestamp.hour*3600 + timestamp.minute*60 + timestamp.second + timestamp.microsecond*1e-6)

            pairs = dict(re.findall(r"\[(\w+)\s*:\s*([^\]]+)\]", subtitle.text))

            altitude.append(float(pairs["altitude"]))
            latitude.append(float(pairs["latitude"]))
            longitude.append(float(pairs["longtitude"]))

        else:
            raise KeyError(f"Warning: No timestamp found in subtitle index {index} with text: {subtitle.text}")

    return MeasurementDataSRT(timestamp=timestamp_seconds, altitude=altitude, latitude=latitude, longitude=longitude)
