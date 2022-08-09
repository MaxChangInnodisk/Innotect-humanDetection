"""AreaDetection.py

This code will allow user to set up their own area,
and display it (or, draw it) at the same time.
The main wi
"""

class AeraDetection(App):
    def __init__(self, depend_labels) -> None:
        super().__init__(depend_labels)
        self.in_area = False
        self.alpha = ALPHA
        self.pnts = []
        self.warn_color = WARN_COLOR
        self.area_color = AREA_COLOR
        self.det_color = DET_COLOR
        self.temp_frame = None
        self.temp_draw = None
        self.temp_pnts = []
    def set_area(self, pnts=None, frame=None):
        if pnts==None and frame.all()==None:
            msg = "Could not capture polygon coordinate, reset again"
            logging.warning(msg)
            raise Exception(msg)
        if pnts==None and frame.all()!=None:
            logging.warning("Detected frame, open the cv windows to collect area coordinate")
            self.temp_frame = frame.copy()
            self.temp_draw = frame.copy()
            cv2.setMouseCallback(CV_WIN, self.add_point)
            while True:
                cv2.putText(self.temp_draw, "Click to define detected area, press q to leave", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow(CV_WIN, self.temp_draw)
                if cv2.waitKey(1)==ord('q'):
                    break
            # cv2.destroyAllWindows()
            self.pnts = self.temp_pnts
        if pnts!=None:
            self.pnts = pnts
    def __call__(self, frame, info):
        # double check
        if self.pnts==None:
            msg = "Could not capture polygon coordinate, reset again"
            logging.warning(msg)
            raise Exception(msg)
        size = frame.shape[:2]
        # draw poly point
        overlay = frame.copy()
        for pnt in self.pnts:
            cv2.circle(overlay, pnt, 3, AREA_COLOR, -1)
        # file poly and make it transparant
        cv2.fillPoly(overlay, pts=[ np.array(self.pnts, np.int32) ], color=AREA_COLOR)
        frame = cv2.addWeighted( frame, ALPHA, overlay, 1-ALPHA, 0 )
        for detection in info["detections"]:
            label = detection['label']
            if label in self.depend_labels:
                # set up trigger points
                self.in_area = False
                x1, y1 = max(int(detection['xmin']), 0), max(int(detection['ymin']), 0)
                x2, y2 = min(int(detection['xmax']), size[1]), min(int(detection['ymax']), size[0])
                dx, dy = x2-x1, y2-y1
                # 121 trigger in area
                steps = np.linspace(0,1,11)
                # only top-left and bottom-right and center
                # steps = np.linspace(0,1,3)
                # only center
                # steps = np.linspace(0.5,0.5,1)
                # center margin 3 step
                steps = np.linspace( 0.3, 0.7, 5 )
                # generate trigger
                x_list = [ x1 + int( dx*step ) for step in steps ]
                y_list = [ y1 + int( dy*step ) for step in steps ]
                triggers = [ ( x, y ) for x in x_list for y in y_list ]
                # detect whether in area
                for pnt in triggers:
                    dist = int(cv2.pointPolygonTest(np.array(self.pnts, np.int32), (pnt), False))
                    if dist==1:
                        self.in_area=True
                # draw bbox
                color = self.warn_color if self.in_area else self.det_color
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, BORDER)
        return frame
    def add_point( self, event, x, y, flags, param):
        # if click down
        if event == cv2.EVENT_LBUTTONDOWN:
            # init
            overlay = self.temp_frame.copy()
            self.temp_draw = self.temp_frame.copy()
            # append points
            self.temp_pnts.append( [ x, y ] )
            print(self.temp_pnts)
            # draw poly point
            for pnt in self.temp_pnts:
                cv2.circle(overlay, pnt, 3, AREA_COLOR, -1)
            # file poly and make it transparant
            cv2.fillPoly(overlay, pts=[np.array(self.temp_pnts, np.int32)], color=AREA_COLOR)
            self.temp_draw = cv2.addWeighted( self.temp_draw, self.alpha, overlay, 1-self.alpha, 0 )