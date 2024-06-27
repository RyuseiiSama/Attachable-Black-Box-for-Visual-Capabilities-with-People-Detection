Hi

(Within workspace directory, with src folder and launch.py)

Build using: colcon build
Source using: source ./install/setup.bash
launch using: ros2 launch launch.py

Yolov5 pt file required, will autodownload if all goes well
The 3 nodes will launch: Camera, interpreter and evaluator.
Topcis published will be /detections and /labelledimg. Labelled img will be the labelled photo including the distance.
