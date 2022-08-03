"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import math
import os, sys, logging
import time
import argparse
from typing import Counter

import cv2
import numpy as np
import pycuda.autoinit
#from nvidia.Desktop.tensorrt_demos.utils import AreaDetection  # This is needed for initializing CUDA driver

from utils.style import BLUE, RED, GREEN, WHITE, KELIN,\
    FONT_SCALE, FONT, LINE, ALPHA
from utils.area_detect import AreaDetection
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME     = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    #　提示詞
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)


    # 導入 camer.py 中的 add_camera_args，程式碼統一管理
    parser = add_camera_args(parser)


    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.5,
        help='set the detection confidence threshold')
    # 需要利用關鍵字 -m yolo3... 去使用 model
    # Default type = string
    # required = True 代表選項不可略過
    # -m 是短參數 
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    #　解析參數
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.
    
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    frame_count = 0
    Track_object = {}
    Track_id =0 
    distance_limit = 60
    
    # draw Area

    # win1, win2 = "Area01", "Area02"
    # win_name1, win_name2 = "Draw Area 01", "Draw Area 02"
    # area1 = AreaDetection(cam, win1, win_name1)
    # area2 = AreaDetection(cam, win2, win_name2, area1)


    while True:
        frame_count += 1
        cur_pnt = []
        tic = time.time()
        img = cam.read()
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0 or img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        
        # out of loop
        
        # Draw Polygon
        # overlay = img.copy()
        # cv2.polylines(overlay, pts=[np.array(area1.save_pnt, np.int32)], isClosed=True, color=BLUE, thickness=3)
        # cv2.polylines(overlay, pts=[np.array(area2.save_pnt, np.int32)], isClosed=True, color=RED, thickness=3)
        # img = cv2.addWeighted( img, 0.1, overlay, 0.9, 0 )

            # in loop
        
            # size = cv2.getTextSize("Alert", FONT, FONT_SCALE, 5)
            # txt_w, txt_h = size[0][0], size[0][1]
            # txt_x, txt_y = int((x_min + x_max - txt_w)/2), int((y_min + y_max)/2)
            # feet_x, feet_y = int((x_min + x_max)/2), int(y_max)
        print("\n\n")
        print("----------------------------")
        print("Frame number:" + str(frame_count))
        print("----------------------------")

        for idx, bb in enumerate(boxes):     

            if (clss[idx]!=0):
                continue

            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            center_x, center_y = int((x_min + x_max)/2), int((y_min + y_max)/2)

            # Append Current Points
            cur_pnt.append((center_x, center_y))

        # For first frame
        if frame_count <= 1:
            # for pt in cur_pnt:
            #     for pt2 in pre_pnt:
            #         distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
            #         if distance < distance_limit:
            #             Track_object[Track_id] = pt
            #             Track_id += 1
            #             print("First {} item ( distance: {})".format(Track_id, distance))
            #         else:
            #             print("Out of dis ({}, {} , distance: {})".format(pt, pt2, distance))

            for pt in cur_pnt:
                Track_object[Track_id] = pt
                Track_id += 1
                
        # The followimng
        else:

            # due to the rule -- you can't change list during loop
            Tracking_object_copy = Track_object.copy()

            cur_pnt_copy = cur_pnt.copy()
            
            for id, pt2 in Tracking_object_copy.items():
                
                object_exist = False

                for pt in cur_pnt_copy:
                    distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                    # update id position
                    if distance < distance_limit:
                        
                        Track_object[id] = pt
                        object_exist = True

                        # if object is exist delete point
                        if pt in cur_pnt:
                            # if object is not exist, remain point
                            cur_pnt.remove(pt)

                        break
                    
                if not object_exist:
                    # pop is remove the key in dictionary, if not key here, do nothing
                    Track_object.pop(id)

            # Add new id founded
            for pt in cur_pnt:
                Track_object[Track_id] = pt
                Track_id += 1
                
        # Show the position
        for id, pt in Track_object.items():
            cv2.circle(img, pt, 5, RED, -1)
            cv2.putText(img, str(id), (pt[0], pt[1]-7), FONT, 1, RED, 2)

        # txt_p = (txt_x, txt_y)

        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        
        # pre_pnt = cur_pnt.copy()

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)                        

def main():
    # 取得外部變數
    args = parse_args()

    # 確認類別數量，錯誤就停止運行
    # from parse_args() "-c" "-category"
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    # 取得來源: 支援 Image, Video ...
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # 取得標籤(字典)
    cls_dict = get_cls_dict(args.category_num)
    # 取得畫圖物件
    vis = BBoxVisualization(cls_dict)
    # 取得模型
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    # 設定CV視窗大小跟標題
    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)

    # 進入迴圈 持續進行辨識
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)

    # 
    cam.release()
    cv2.destroyAllWindows()
    sys.exit()

# 避免 trt_yolo.py 在 import 階段，不小心誤執行
if __name__ == '__main__':
    main()

'''
    img = cam.read()
    ORG_FRAME = img.copy()
    DRAW_FRAME = ORG_FRAME.copy()
    cv2.imshow(WINDOW_NAME, DRAW_FRAME)
    # user draw
    # wait untill user draw
    # save point as list [x1, y1 ...]
    cv2.setMouseCallback(WINDOW_NAME, add_point)

    # show frame
    while True:
        cv2.imshow(WINDOW_NAME, DRAW_FRAME)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        elif key in [ ord("c"), ord("C") ]:
            DRAW_FRAME = ORG_FRAME.copy()
            TEMP_POINT = []
    # cv2.destroyWindow(WINDOW_NAME)
'''

'''
def add_point(event, x, y, flags, param):
    """add detection area point via click"""
    global DRAW_FRAME
    # if click down
    if event == cv2.EVENT_LBUTTONDOWN:
        # init
        RED = (255, 0, 0)
        ALPHA = 0.5
        overlay = ORG_FRAME.copy()
        DRAW_FRAME = ORG_FRAME.copy()
        
        # append points
        TEMP_POINT.append( [ x, y ] )
        # draw poly point
        for pnt in TEMP_POINT:
            cv2.circle(overlay, pnt, 3, RED, -1)
        # file poly and make it transparant
        cv2.fillPoly(overlay, pts=[np.array(TEMP_POINT, np.int32)], color=RED)
        DRAW_FRAME = cv2.addWeighted( DRAW_FRAME, ALPHA, overlay, 1-ALPHA, 0 )
'''