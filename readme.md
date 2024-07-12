
# Project Description
This project was done during my internship. This project mainly features the use of **ROS2**, **OpenCV**, **YOLOv5** by Ultralytics and some **Bash** scripts.

This project is creating a prototype attachable instrument to enable visuals on any ground control robot.

# Contents
- Creation
  - [Hardware](##hardware)
  - [Software](##software)
- [Installation]()
- [Usage]()
- [Additional Features]()


# Creation

This section will be split into 2 components; hardware and software.

_Hardware_ will feature the electrical components and the design of the case.

_Software_ will feature the internal architecture of the source code.


## Hardware


(Within workspace directory, with src folder and launch.py)
Build using: colcon build
Source using: source ./install/setup.bash
launch using: ros2 launch launch.py

Yolov5 pt file required, will autodownload if all goes well

The 3 nodes will launch: Camera, interpreter and evaluator.
Topics published will be /detections and /labelledimg. Labelled img will be the labelled photo including the distance.
