"""
TODO: 
"""
import sys
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import argparse

from .chessboard_data import ChessboardData

parser = argparse.ArgumentParser(prog="Camera Calibration", description="TODO: description", epilog="TODO: text at the bottom of script help message")

parser.add_argument('path', help="Path to the directory containing calibration images or a video file")    
parser.add_argument('-v', '--visualise', action='store_true', help="Visualise the detected corners on the chessboard images using OpenCV")
parser.add_argument('-q', '--quiet', action='store_true', help="Disable rerun visualisation")

args = parser.parse_args()

print("Path to calibration data: ", args.path)

if not args.path:
    print("Error: Path to calibration data is required.")
    parser.print_help()
    sys.exit(1) 

if not args.quiet:
    rr.init("Camera Calibration", spawn=True)

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="World"),
        rrb.Spatial2DView(origin="/world/camera", name="Camera", contents=["/world/**"]),
    )
    
    rr.script_add_args(parser)
    rr.script_setup(args, "Camera Calibration", default_blueprint=blueprint)

chessboard_data = ChessboardData(pattern_size=(10,7), square_size=0.022)

chessboard_data.calibrate_camera(args.path, visualise=args.visualise, export=False, pattern_size=(10,7))

# # Get the directory where the current Python script is located
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Build absolute path relative to this script
# data_path = os.path.join(script_dir, '../../data/images/*.JPG')

# calibrate_camera(data_path, visualise=False, export=False, pattern_size=(10,7))


def calc_field_of_view(camera):
    assert camera.camera_matrix.shape == (3, 3)
    assert camera.camera_matrix.dtype == np.float64

    return


def body_to_camera(Tnb, Tbc):
    # Pose Camera::bodyToCamera(const Pose & Tnb) const
# {
#     // Tnc = Tnb*Tbc
#     return Tnb*Tbc;
# }
    return Tnb * Tbc

def world_to_vector(rPNn, Tnb, Tbc):

    Tnc = body_to_camera(Tnb, Tbc)
    rPCc = Tnc.rotation_matrix @ (rPNn - Tnc.translation_vector)
    uPCc = rPCc / np.linalg.norm(rPCc)

    return uPCc