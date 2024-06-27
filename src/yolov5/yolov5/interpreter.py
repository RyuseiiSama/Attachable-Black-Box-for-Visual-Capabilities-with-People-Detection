import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import pyrealsense2 as rs2


def clean_str(string:str):
   ''' cleans string of ',' and spaces on the ends '''
   if string[0] == ',' or string[0] == ' ':
      return clean_str(string[1:])
   elif string[-1] == ',' or string[-1] == ' ':
      return clean_str(string[:-1])
   else:
      return string
class Interpreter(Node):
    def __init__(self):
      super().__init__('interpreter')
      self.subscription = self.create_subscription(String,'detections',self.process_msg,10)
      self.subscription 

      # For distance calculation
      self.bridge = CvBridge()
      self.sub = self.create_subscription(msg_Image, 'camera/camera/depth/image_rect_raw', self.image_handler, 10)
      self.sub_info = self.create_subscription(CameraInfo, 'camera/camera/depth/camera_info', self.imageDepthInfoCallback, 10)
      self.color_img = self.create_subscription(msg_Image,'/camera/camera/color/image_raw',self.color,10)
      self.picam1 = self.create_subscription(msg_Image,'/image_raw1',self.rpicam1,10)
      self.picam2 = self.create_subscription(msg_Image,'/image_raw2',self.rpicam2,10)
      self.publisher = self.create_publisher(msg_Image, 'labelled_img' ,10)
      self.combined = self.create_publisher(msg_Image, 'combined_img', 10)
      self.intrinsics = None
      self.pix_grade = None
      self.img = None
      self.piimg1 = None
      self.piimg2 = None
      import pyrealsense2 as rs2
      if (not hasattr(rs2, 'intrinsics')):
        import pyrealsense2.pyrealsense2 as rs2
      self.bboxes = []


    def color(self,img):
       self.get_logger().info("received an image")
       self.img = img
    def rpicam1(self,img): # Capture rpicam1 image
      self.piimg1 = img
    def rpicam2(self,img): # Capture rpicam2 image
      self.piimg2 = img
      
    def process_msg(self,msg):
      ''' Returns a list of all bounding boxes'''
      unprocessed = msg.data

      ## Processing ##
      split = unprocessed.split(']')
      for i in range(len(split)):
        if split == []:
           self.bboxes = []
        
        split[i] = split[i].replace('[','')
        if split[i] == '':
           continue
        split[i] = clean_str(split[i])
      while '' in split:
         split.remove('')
      bboxes = [list(map(int,x.split(','))) for x in split]
      self.bboxes = bboxes
    

    def find_centre(self,bbox):
       ''' Returns the centre of each bbox'''
       i = bbox
       x1 = i[0]
       y1 = i[1]
       x2 = i[2]
       y2 = i[3]
       centre = (int((x1+x2)/2),int((y1+y2)/2))
       return centre

    def image_handler(self,data):
       
       # If no detections
       if self.bboxes == []:
          print("Nothing Detected")
          if not self.img:
             return
          self.publisher.publish(self.img)
          self.combined_img(self.img)
          return
       
       self.cimg = self.bridge.imgmsg_to_cv2(self.img,self.img.encoding)
       self.dimg = self.bridge.imgmsg_to_cv2(data, data.encoding)
       centres = []
       for i in self.bboxes:
          centres.append(self.find_centre(i))  # Get a list of centres for detected bounding box
       depths = self.imageDepthCallback(self.dimg,centres)
       for x in range(len(depths)):
          # Write distance 
          cv2.putText(self.cimg,f'd={round(depths[x],3)}m',(self.bboxes[x][0],self.bboxes[x][1]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2,cv2.LINE_AA)
          # Draw lines from centre
          cv2.line(self.cimg,(320,240),centres[x],color = (255,255,255))

          # Draw Bounding box
          cv2.rectangle(self.cimg, (self.bboxes[x][0], self.bboxes[x][1]), (self.bboxes[x][2],self.bboxes[x][3]), (255,255,255) , 1)
       final_img = self.bridge.cv2_to_imgmsg(self.cimg,self.img.encoding)
       self.publisher.publish(final_img)
       self.combined_img(final_img)
    
    def combined_img(self,labelled_img):
        if not self.piimg1 or not self.piimg2:
            self.get_logger().info("SMLHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            return
        piimg1 = self.bridge.imgmsg_to_cv2(self.piimg1,self.piimg1.encoding)
        self.get_logger().info(f"piimg1 IS A {self.piimg1.encoding}")
        self.get_logger().info(f"piimg2 IS A {self.piimg2.encoding}")
        self.get_logger().info(f"label IS A {labelled_img.encoding}")
        piimg2 = self.bridge.imgmsg_to_cv2(self.piimg2,self.piimg2.encoding)
        depthcamimg = self.bridge.imgmsg_to_cv2(labelled_img,labelled_img.encoding)

        bottom_row_width = piimg1.shape[1] + piimg2.shape[1]
        output_height = piimg1.shape[0] + depthcamimg.shape[0]
        temp_image = np.zeros((output_height,bottom_row_width,3),dtype=np.uint8)

        # Place top image 
        temp_image[:depthcamimg.shape[0],depthcamimg.shape[1]//2:depthcamimg.shape[1]//2 + depthcamimg.shape[1]] = depthcamimg

        # Place bottom images
        btm = cv2.hconcat([piimg1,piimg2])
        temp_image[depthcamimg.shape[0]:output_height,:bottom_row_width] = btm

        # pyramid_img = cv2.vconcat([cv2.hconcat([piimg1,piimg2]),depthcamimg])
        self.combined.publish(self.bridge.cv2_to_imgmsg(temp_image,"rgb8"))
	

    def imageDepthCallback(self, cv_image, centres):
        depths = []
        for i in centres:
          try:
              # pick one pixel among all the pixels with the closest range:
              # indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
              ################
              pix = i
              # line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (pix[0], pix[1], cv_image[pix[1], pix[0]])
              print(cv_image.shape)
              depth = cv_image[pix[1], pix[0]]/1000
              depths.append(depth)
              if self.intrinsics:
                  result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                  # line += '  Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2])
              if (not self.pix_grade is None):
                  line += ' Grade: %2d' % self.pix_grade
              # self.publisher.publish(msg)

          except CvBridgeError as e:
              print(e)
              return
          except ValueError as e:
              return
        print(depths)
        return depths
      
    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]
        except CvBridgeError as e:
            print(e)
            return
      
      
def main():
    rclpy.init(args=None)
    interpreter = Interpreter()
    rclpy.spin(interpreter)


if __name__ == '__main__':
   main()
