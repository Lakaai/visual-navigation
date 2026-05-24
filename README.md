# Assignment 2 Visual Navigation

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cd build
```
```bash
./a2 --calibrate ../data/outdoor/config.xml
```

```bash
ninja && ./a2 --scenario=4 --interactive=0 --export ../data/outdoor/flight.MOV
```

## Command Line Arugments

The --interactive parameter passed to the command line controls the action of the mouse interactor
in the visualisation window as described below:

• If --interactive=0 (default value), the mouse interactor is not enabled and the visualisation
of each frame in the video sequence is rendered without ever being blocked on user input.

• If --interactive=1, the mouse interactor is enabled on the visualisation on the last frame of
the video sequence only, enabling the user to explore the final map solution in the right pane
through the pan and zoom controls, but blocks the application from terminating.

• If --interactive=2, the mouse interactor is enabled on the visualisation on every frame of
the video sequence, blocking the application on user input at each frame. This is useful for
troubleshooting issues during the first few frames.


# Notes
Template functions are used for autodiff. 