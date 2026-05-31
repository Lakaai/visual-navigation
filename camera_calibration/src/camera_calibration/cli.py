"""
TODO: 
"""
import sys
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import argparse

from .chessboard_data import ChessboardData

def parse_pattern_size_tuple(value):
    try:
        x, y = map(int, value.split('x'))
        return (x, y)
    except ValueError:
        raise argparse.ArgumentTypeError("Pattern size must be in 'WxH' format (e.g. 10x7)")

def main():
    parser = argparse.ArgumentParser(prog="Camera Calibration", description="TODO: description", epilog="TODO: text at the bottom of script help message")

    parser.add_argument('path', help="Path to the directory containing calibration images or a video file")    
    parser.add_argument('-v', '--visualise', action='store_true', help="Visualise the detected corners on the chessboard images using OpenCV")
    parser.add_argument('-e', '--export', action='store_true', help="Export the calibration results to a JSON file")
    parser.add_argument('-q', '--quiet', action='store_true', help="Disable rerun visualisation")
    parser.add_argument('--pattern-size', type=parse_pattern_size_tuple, default=(10,7), help="The number of internal corners per chessboard row and column, default is (10, 7)")
    parser.add_argument('--square-size', type=float, default=0.022, help="The size of the squares on the chessboard in meters, default is 0.022m (22mm)")

    args = parser.parse_args()

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

    chessboard_data = ChessboardData(pattern_size=args.pattern_size, square_size=args.square_size)

    chessboard_data.calibrate_camera(args.path, visualise=args.visualise, export=args.export)

if __name__ == "__main__":
    main()