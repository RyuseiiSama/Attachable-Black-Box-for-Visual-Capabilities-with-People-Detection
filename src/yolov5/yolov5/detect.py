# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path


import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

#  ---------------------- ROS Imports -------------------------
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError



class Evaluator(Node):
    def __init__(self):
        super().__init__('evaluator')
        self.subscription = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.evaluateee,
            10)
        self.publisher = self.create_publisher(String,'detections',10)
        self.subscription  # prevent unused variable warning
        self.webcam = False
        # Directories
        self.weights=ROOT / "yolov5s.pt"  # model path or triton URL
        self.source=ROOT / "data/images"  # file/dir/URL/glob/screen/0(webcam)
        self.data=ROOT / "data/coco128.yaml"  # dataset.yaml path
        self.imgsz=(480, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device="0"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.source = str(self.source)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        # Dataloader
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        print("Model Loaded, waiting for topic")


    # @smart_inference_mode
    def evaluateee(self,msg):
        img = msg
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(img,desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)
        self.get_logger().info("Generating Results...")

        im = image # Feed numpy image here
        im0 = image
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if self.model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
            im = torch.permute(im,(0,3,1,2))

        # Inference
        with self.dt[1]:
            visualize = False
            if self.model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = self.model(image, augment=self.augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, self.model(image, augment=self.augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = self.model(im, augment=self.augment, visualize=visualize)
        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        to_send = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            # s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = self.names[c] if self.hide_conf else f"{self.names[c]}"
                    confidence = float(conf)
                    if c == 0 and conf>0.8:
                        temp = []
                        for i in xyxy:  # xyxy is bbox loc, conf is confidence, cls is class. cls 0 is person
                            temp.append(int(i)) 
                        to_send.append(temp)
        send = String()
        send.data = str(to_send)
        self.publisher.publish(send)

                

        


          


def main():
    rclpy.init(args=None)
    evaluator = Evaluator()

    rclpy.spin(evaluator)

    evaluator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
