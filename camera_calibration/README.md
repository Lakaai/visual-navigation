from the root of camera calibration dir run

```bash
pip install -e .
```

Run camera calibration application in quite mode with square size 22mm and pattern size 10 columns by 7 rows, also export result with -e flag 
```bash
calibrate -qe /home/luke/MCHA4400/data/outdoor/calibration.MOV --square-size 0.022 --pattern-size 10x7
```

# TODO: 
# Get the directory where the current Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build absolute path relative to this script
data_path = os.path.join(script_dir, '../../data/images/*.JPG')

# Future work:
potentially check every calibration image has same resolution before proceeding, if not then throw error. 