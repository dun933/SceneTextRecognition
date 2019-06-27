# Scene Text Recognition

import nms.nms
import numpy as np
import sys
import cv2

class SceneTextImage:
    east = 'frozen_east_text_detection.pb'
    min_confidence = 0.5
    reshapeW = 320
    reshapeH = 320
    padding = 0.1
    nms_threshold = 0.3

    def __init__(self, imgFile = "", **kwargs):
        self.init(imgFile, **kwargs)

    def init(self, imgFile = "", **kwargs):
        if imgFile:
            self.imageFile = imgFile
            self.winName = "Detected:" + self.imageFile
        if 'east' in kwargs:
            self.east = kwargs['east']
        if 'min_confidence' in kwargs:
            self.w = kwargs['min_confidence']
        if 'w' in kwargs:
            self.w = kwargs['reshapeW']
        if 'h' in kwargs:
            self.w = kwargs['reshapeH']
        if 'padding' in kwargs:
            self.w = kwargs['padding']
        if 'nms_threshold' in kwargs:
            self.w = kwargs['nms_threshold']

    def translateRbox2Coord(self, x, y, t, r, b, l, a):
        cos = np.cos(a)
        sin = np.sin(a)
        topleft = (x - l * cos - t * sin, y - l * sin + t * cos)
        topright = (x + r * cos - t * sin, y + r * sin + t * cos)
        bottomright = (x + r * cos + b * sin, y + r * sin - b * cos)
        bottomleft = (x - l * cos + b * sin, y - l * sin - b * cos)
        return (topleft, topright, bottomright, bottomleft)

    def translate2CenteredRbox(self, x, y, t, r, b, l, a):
        (topleft, topright, bottomright, bottomleft) = self.translateRbox2Coord(x, y, t, r, b, l, a)
        cx = (topleft[0] + bottomright[0]) / 2
        cy = (topleft[1] + bottomright[1]) / 2
        w, h = l + r, t + b
        deg = a / np.pi * 180
        return ((cx, cy), (w, h), deg)

    def decode_predictions(self, scores, geometry):
        """
        coordiantion maps with (x, y) where x = width / 4, y = height / 4
        scores.shape = (1, 1, y, x)
        geometry.shape = (1, 5, y, x)
        
        geometry[0, 0] : distance to TOP border
        geometry[0, 1] : distance to RIGHT border
        geometry[0, 2] : distance to BOTTOM border
        geometry[0, 3] : distance to LEFT border
        geometry[0, 4] : rotation angle
        """
        (yRange, xRange) = scores.shape[2:4]
        polys = []
        confidences = []

        for y in range(yRange):
            s = scores[0, 0, y] # scores
            t = geometry[0, 0, y] # distance to TOP border
            r = geometry[0, 1, y] # distance to RIGHT border
            b = geometry[0, 2, y] # distance to BOTTOM border
            l = geometry[0, 3, y] # distance to LEFT border
            a = geometry[0, 4, y] # rotation anlge
     
            for x in range(xRange):
                if s[x] < self.min_confidence:
                    continue
     
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (ox, oy) = (x * 4.0, y * 4.0)
     
                polys.append(self.translateRbox2Coord(ox, oy, t[x], r[x], b[x], l[x], a[x]))
                confidences.append(s[x])

        return (polys, confidences)

    def detect(self):
        self.original_image = self.imread(self.imageFile)
        # preserve original image
        self.image = self.original_image.copy()
        h, w = self.image.shape[:2]

        reshape_ratio = np.array((w / float(self.reshapeW), h / float(self.reshapeH)))

        reshaped_image = cv2.resize(self.image, (self.reshapeW, self.reshapeH))

        # define the two output layer names for the EAST detector model that
        # we are interested in -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        net = cv2.dnn.readNet(self.east)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(reshaped_image, 1.0, (self.reshapeW, self.reshapeH),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (polys, confidences) = self.decode_predictions(scores, geometry)

        indexes = nms.nms.polygons(polys, confidences, nms_threshold=self.nms_threshold)

        if indexes:
            self.nmspolys = np.array(np.array(polys)[indexes] * reshape_ratio, np.int32)
            self.nmsscores = np.array(confidences)[indexes]
        else:
            self.nmspolys = []
        return len(self.nmspolys)

    def showAll(self):
        outimg_all = self.image.copy()
        len_polys = len(self.nmspolys)
        escaped = False
        for i in range(len_polys):
            if not escaped:
                keycode = self.show(i)
                if keycode == 27:
                    escaped = True
            poly = self.nmspolys[i]
            cv2.polylines(outimg_all, [poly], True, (0,0,255), 1)
            cv2.putText(outimg_all, f"{self.nmsscores[i]*100:.2f}%", (poly[0,0], poly[0,1]+18), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)

        cv2.imshow(self.winName, outimg_all)
        cv2.setWindowTitle(self.winName, "Detected All")
        cv2.waitKey(0)

    def show(self, i=-1):
        if i < 0:
            self.showAll()
        else:
            poly = self.nmspolys[i]
            len_polys = len(self.nmspolys)
            outimg = self.image.copy()
            cv2.polylines(outimg, [poly], True, (0,0,255), 1)
            cv2.putText(outimg, f"{self.nmsscores[i]*100:.2f}%", (poly[0,0], poly[0,1]+18), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)
            cv2.imshow(self.winName, outimg)
            cv2.setWindowTitle(self.winName, f"Detected ({i+1}/{len_polys})")
            return cv2.waitKey(0)
    def imread(self, imgPath):
        img = cv2.imread(imgPath)
        if img is None:
            bytes = bytearray(open(imgPath, 'rb').read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return img

if __name__ == "__main__":
    if len(sys.argv) > 1:
        imageFile = sys.argv[1]
    else:
        imageFile = "images/car_wash.png"
    sample = SceneTextImage(imageFile)
    detected = sample.detect()
    print(f"Detected {detected} regions")
    sample.show()