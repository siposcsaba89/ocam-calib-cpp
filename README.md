# ocam-calib-cpp
Omnidirectional camera calibration implementation c++

## Dependencies

* OpenCV with contrib (omni module) version >= 3.4
* ceres
* Eigen3

## ocam-fisheye-calib

```
his is a camera calibration sample.
Usage: calibration
     -w=<board_width>         # the number of inner corners per one of board dimension
     -h=<board_height>        # the number of inner corners per another board dimension
     ...
     [input_data]             # input data, one of the following:
                              #  - text file with a list of the images of the board
                              #    the text file can be generated with imagelist_creator
                              #  - name of video file with a video of the board
                              # if input_data not specified, a live view from the camera is used

```

## Example params

```
./ocam-fisheye-calib -w=8 -h=5  images_cam2.yaml
```

