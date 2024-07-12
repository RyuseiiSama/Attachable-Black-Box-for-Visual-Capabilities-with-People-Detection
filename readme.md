
# Project Description
This project was done during my internship. This project mainly features the use of **ROS2**, **OpenCV**, **YOLOv5** by [Ultralytics](https://github.com/ultralytics/yolov5) (Initially was [YOLACT](https://github.com/dbolya/yolact), in commt history) and some **Bash** scripts.

This project is creating a prototype attachable instrument to enable visuals on any ground control robot.

# Contents
- Creation
  - [Hardware](#hardware)
  - [Software](#software)
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
- Servo Motor

### Casing 
The casing was designed using Siemens NX 12. NX Modeling was used to design the box itself, NX Assembly was used to see if other components would fit into the case (to reduce design iterations). Case was then 3D printed using PLA on the Ultimaker Cura.
![1](https://github.com/user-attachments/assets/2e503561-bb2a-403d-b52d-1286d6d22f28)![2](https://github.com/user-attachments/assets/ab413266-d689-4042-ab49-e0fca751c8b1)

The Depth camera is mounted on the servo motor using a custom-made holder as well. This allows the depth camera to be rotated both manually and automatically. See [Additional Features](#additional-features)


## Software
### Structure
Using ROS2 as my main framework:
![image](https://github.com/user-attachments/assets/b11856ac-6398-45ee-8ac1-b1ea334a3c7c)

Nodes:

/**camera** - Published by Intel RealSense ROS2 Package

/**evaluator** - Adapted YOLOv5 detect.py functioning as a node, publishes detected humans in topic

/**interpreter** - Annotates depth camera stream with 1) Human bounding box and 2) Distance to human. Then combines 3 camera stream (2x rpicam + depth camera) to publish to /combined_img
 
### Object Detection
Adpating the Yolov5 repository by Ultralytics, configuring ```detect.py``` to be featured as a ROS2 Node (evaluator). Set default values for classes detected to be only id 0 (human) and confidence level.

### Intel RealSense 
Mostly adopting the Intel RealSense ROS2 Package to obtain RGB and Depth Stream.

### Rpi Camera
Used [gscam2](https://github.com/clydemcqueen/gscam2) to launch a curated gstreamer pipeline which relied on nvidia's nvarguscamerasrc plugin to stream CSI cameras from the jetson board

### Web Video Server (Not included in framework)
To view all image streams via http, I included the [Web Video Server Package](https://wiki.ros.org/web_video_server) (i built from source) which publishes **ALL** camera streams to your localhost:8080. This web server was only accessible if only on the same network as the Jetson Xavier NX, which I explained in the next section.

### Others

This section includes:
- Steps taken to optimize computing power on the Jetson Xavier NX
- Making it as plug-and-play-able as possible

1) Launching in multi-user.target mode

To reduce Graphical processing and reduce background services that boot up (lookup:bloat)

``` sudo systemctl set-default multi-user.target ```

2) Launch Script ```launch.sh```

Created a launch script that will launch ``` launch.py ``` and 3 other nodes (2x gscam and 1x web_video_server) as background processes to start them up in parallel

3) **Launching Jetson Xavier NX's hotspot on startup**

Okay, this part was a little beyond me, so I will explain the steps that i initially went through. Bear in mind I wish to keep my multi-user.target target.

[First]: Using ```nmcli```

Using ```  nmcli connection modify Hotspot connection.autoconnect true ``` , I thought I had set the default for the hotspot connection to be up on startup. I even disabled autoconnection for all other possible WiFi connections.

**Problem**: Hotspot detected by External workstation, could not connect at all. Could only work if i logged in (manully) with username and password on command line.

[Second]: Using ```crontab```

I included ``` nmcli connection modify Hotspot connection.autoconnect true ``` into the mentioned ```launch.sh```, and had ```launch.sh``` run at reboot, i.e:

```crontab -e```

Inserted:

```@reboot /bin/bash /home/user1/Desktop/launch.sh```

**Problem**: nmcli line faced errors as NetworkManager did not seem to start up yet. Could not tell as I (then) did not understand how to view debug lines in background processes.

[Third]: Using ```expect``` and launching it as a service
Expect is an older solution, but I decided to give it a shot.

In Pseudo Code, my expect code for ```login.exp``` (not included in repo for now) was as follows:

- spawn ```ssh user1@localhost```
- expect "assword" -> give password
- spawn ```sudo nmcli connection up Hotspot```
- expect "assword" again -> give password
- expect "connection successfully" -> logout

This was then converted into a service, whereby this service would be called AFTER= ```multi-user.target``` and ```NetworkManagerOnline.target``` was hit.

**Problem**: The service seems to work once or twice unreliably, and seemed to stop after successfully logging into local ssh. This may be due to my inexperience working with BOTH expect and Linux services.

[Fourth]: Just using the ssh with ```expect```

This was chanced upon unexpectedly. As mentioned in my first problem, autoconnect only worked after logging in.

Thus:

```login.exp``` service to login through ssh user1@localhost, then logout right away to achieve the same effect. 

**_PROBLEM SOLVED_**


# Installation
Installation only for the ROS2 Packages, including those sourced externally and those created by me.

## Clone this workspace
``` git clone <repo_link> ```

## Build
(Within workspace directory, next to src folder and launch.py)

``` colcon build ```

Source the workspace

``` source ./install/setup.bash ```
launch using: ros2 launch launch.py

Yolov5 pt file required, will autodownload if all goes well

The 3 nodes will launch: Camera, interpreter and evaluator.
Topics published will be /detections and /labelledimg. Labelled img will be the labelled photo including the distance.

# Additional Features
