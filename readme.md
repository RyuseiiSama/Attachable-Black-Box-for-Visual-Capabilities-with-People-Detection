
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

### Wiring
![image](https://github.com/user-attachments/assets/51227db9-7c74-45a7-b256-3f544db5ba76)

Components:
- Nvidia Jetson Xavier NX
- Intel RealSense D455 Depth Camera
- 2x RpiCamera v2.1
- 3S Lipo Battery 2000Mah
- DC-DC Step up voltage regulator 5A

### Casing 
The casing was designed using Siemens NX 12. NX Modeling was used to design the box itself, NX Assembly was used to see if other components would fit into the case (to reduce design iterations). Case was then 3D printed using PLA on the Ultimaker Cura.
![1](https://github.com/user-attachments/assets/2e503561-bb2a-403d-b52d-1286d6d22f28)![2](https://github.com/user-attachments/assets/ab413266-d689-4042-ab49-e0fca751c8b1)

The Depth camera is mounted on the servo motor using a custom-made holder as well. This allows the depth camera to be rotated both manually and automatically. See [Additional Features](#additional-features)


## Software

(Within workspace directory, with src folder and launch.py)
Build using: colcon build
Source using: source ./install/setup.bash
launch using: ros2 launch launch.py

Yolov5 pt file required, will autodownload if all goes well

The 3 nodes will launch: Camera, interpreter and evaluator.
Topics published will be /detections and /labelledimg. Labelled img will be the labelled photo including the distance.

# Additional Features
