"""area_detection.py

The AreaDetecion class implements drawing of area detection, 
and also display the previous area region on screen with different color.
Also including 'openwindow' and 'destroywindow'.
"""

import cv2
import numpy as np

from utils.display import open_window 
from utils.style import BLUE, RED, GREEN, WHITE, KELIN,\
    FONT_SCALE, FONT, LINE, ALPHA

class AreaDetection:
    """Draw the area and display it

    Arguments
            cam: RTSP, from class Camera()
            win: window such as 'Area 1' or 'Area 2'
            win_name: text on the top of window
            other_area = neccessary if you want more than one area
    """
    def __init__(self, cam, win, win_name, other_area=None):
        # frame
        self.isOther = False
        self.org_frame = None
        self.draw_frame = None
        self.overlay = None
        self._setFrame(cam, other_area)
        # win info
        self.img_height = cam.img_height
        self.img_width = cam.img_width
        self.win = win
        self.win_name = win_name
        # save area points as [(x1,y1),(x2,y2)...]
        self.save_pnt = []
        self._drawArea()
    
    def _setFrame(self, cam, other_area=None):
        if other_area:
            self.isOther = True
            self.org_frame = other_area.org_frame
            self.overlay = self.org_frame.copy()
            self.other_pnt = other_area.save_pnt
            cv2.fillPoly(self.overlay, pts=[np.array(other_area.save_pnt, np.int32)], color=KELIN)
            self.org_frame = cv2.addWeighted(self.org_frame, ALPHA, self.overlay, 1-ALPHA, 0)  
            self.draw_frame = self.org_frame.copy()
            return self.isOther, self.org_frame, self.draw_frame, self.overlay
        else:
            self.org_frame = cam.read()
            self.draw_frame = self.org_frame.copy()
            self.overlay = self.org_frame.copy()
            return self.org_frame, self.draw_frame, self.overlay


    def _drawArea(self):
        """setMouseCallback event"""
        open_window(self.win, self.win_name, self.img_width, self.img_height)
        self.remind(self.draw_frame, "Q=continue, C=cleanPoint, H=help")
        cv2.imshow(self.win, self.draw_frame)
        cv2.setMouseCallback(self.win, self.add_pnt)
        self.user()
        cv2.destroyWindow(self.win)

    def add_pnt(self, event, x, y, flags, param):
        """Record the points, refresh it if too much"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.save_pnt) < 8:
                # append point position
                self.refresh_window(win_only=True)
                self.save_pnt.append([x, y])
                for pnt in self.save_pnt:
                    cv2.circle(self.overlay, pnt, 5, BLUE, -1)
                cv2.fillPoly(self.overlay, pts=[np.array(self.save_pnt, np.int32)], color=KELIN)
                self.draw_frame = cv2.addWeighted(self.draw_frame, ALPHA, self.overlay, 1-ALPHA, 0)
            else:
                # refresh frame if not satisfy
                self.refresh_window()
                self.remind(self.draw_frame, "Max points 9")
        if event == cv2.EVENT_MBUTTONDOWN:
            if len(self.save_pnt)!=0:
                self.refresh_window(win_only=True)
                self.save_pnt.pop()
                for pnt in self.save_pnt:
                    cv2.circle(self.overlay, pnt, 5, BLUE, -1)
                cv2.fillPoly(self.overlay, pts=[np.array(self.save_pnt, np.int32)], color=KELIN)
                self.draw_frame = cv2.addWeighted(self.draw_frame, ALPHA, self.overlay, 1-ALPHA, 0)
            else:
                self.refresh_window()
                self.remind(self.draw_frame, "No point")
            
    def user(self):
        """press bottom Q, C, H and get result"""
        while True:
            cv2.imshow(self.win, self.draw_frame)
            key = cv2.waitKey(1)
            if key == ord('Q') or key == ord('q'):
                if len(self.save_pnt) < 3:
                    self.refresh_window()
                    self.remind(self.draw_frame, "At least 3 points")
                else:
                    break
            if key == ord('C') or key == ord('c'):
                    self.refresh_window()
                    self.remind(self.draw_frame, "Clean all points") 
            if key == ord('H') or key == ord('h'):
                    self.refresh_window()
                    self.remind(self.draw_frame, "L click add, M click remove")

    def refresh_window(self, win_only=False):
        """refresh window, win_only=True if you want to clear win only"""
        self.draw_frame = self.org_frame.copy()
        self.overlay = self.org_frame.copy()
        if not win_only:
            self.save_pnt = []

    def remind(self, frame ,txt):
        """remind user what happened now at topleft screen"""
        cv2.putText(frame, txt, (10, 80), FONT, FONT_SCALE, RED, 5, LINE)
        cv2.putText(frame, txt, (10, 80), FONT, FONT_SCALE, WHITE, 3, LINE)
        cv2.putText(frame, txt, (10, 80), FONT, FONT_SCALE, RED, 1, LINE)
        # Note: the number 5, 3, 1 means thickness

'''
class Tracker(object):

    def __init__(self):
        self.cur_center_pnts = []
        self.pre_center_pnts = []
        self.track_object = {}
        self.track_id = 0
        self.frame_count = 0
        self._tracker()
        
    
    def _tracker(self, img , clss, boxes):
            self.frame_count += 1
            for idx, bb in enumerate(boxes):
                    
                    if clss[idx]!=0:
                        break
                    x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
                    center_x, center_y = int((x_min + x_max)/2), int((y_min + y_max)/2)

                    # size = cv2.getTextSize("Alert", FONT, FONT_SCALE, 5)
                    # txt_w, txt_h = size[0][0], size[0][1]
                    # txt_x, txt_y = int((x_min + x_max - txt_w)/2), int((y_min + y_max)/2)
                    # feet_x, feet_y = int((x_min + x_max)/2), int(y_max)

                    self.cur_center_pnts.append((center_x, center_y))

                    if f <= 2:
                        for pt in center_cur_pnts:
                            for pt2 in CENTER_PRE_PNT:
                                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                                if distance < 40:
                                    TRACK_OBJECT[TRACK_ID] = pt 
                                    TRACK_ID += 1 
                    else:
                        for pt in center_cur_pnts:
                            for object_id, pt2 in TRACK_OBJECT.items():
                                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                                if distance < 40:
                                    TRACK_OBJECT[object_id] = pt
                                    

                    for object_id, pt in TRACK_OBJECT.items():
                        cv2.putText(img, str(object_id), txt_p, FONT, FONT_SCALE, RED, 5, LINE)
'''