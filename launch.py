from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir

def generate_launch_description():
  return LaunchDescription([
    IncludeLaunchDescription(
            PythonLaunchDescriptionSource(['/home/ddq2/tester_ws/src', '/realsense2_camera/launch/rs_launch.py']),
            launch_arguments={'depth_module.depth_profile': '640x480x30','rgb_camera.color_profile': '640x480x30'}.items()
    ),
    Node(
      package='yolact',
      namespace='',
      executable='evaluator',
      name='evaluator2'
    ),
    Node(
      package='yolact',
      namespace='',
      executable='interpreter',
      name='interpreter'
    )
    

  ])
