from the root of camera calibration dir run

```bash
pip install -e .
```


# Get the directory where the current Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build absolute path relative to this script
data_path = os.path.join(script_dir, '../../data/images/*.JPG')