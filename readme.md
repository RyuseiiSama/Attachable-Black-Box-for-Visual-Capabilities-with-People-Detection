Hi

(Within workspace directory, with src folder and launch.py)
Build using: colcon build
Source using: source ./install/setup.bash
launch using: ros2 launch launch.py

The 3 nodes will launche: Camera, interpreter and evaluator.
Topcis published will be /detections and /labelledimg. Labelled img will be the labelled photo including the distance.
