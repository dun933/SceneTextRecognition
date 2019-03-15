# Scene Text Recognition

from imutils.video import VideoStream
from imutils.video import FPS
import nms.nms
import numpy as np
import sys
import time
import cv2
import SceneTextImage

class SceneTextVideo(SceneTextImage.SceneTextImage):
    east = 'frozen_east_text_detection.pb'
    min_confidence = 0.5
    reshapeW = 320
    reshapeH = 320
    padding = 0.1
    nms_threshold = 0.3
    frameskip = True

    def __init__(self, videoFile = "", **kwargs):
        self.init(videoFile, **kwargs)

    def init(self, videoFile = "", **kwargs):
        if videoFile:
            self.videoFile = videoFile
            self.winName = "Detected:" + self.videoFile
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
        if 'frameskip' in kwargs:
            self.frameskip = kwargs['frameskip']

    def detect(self):
        # define the two output layer names for the EAST detector model that
        # we are interested in -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        net = cv2.dnn.readNet(self.east)

        # if a video path was not supplied, grab the reference to the web cam
        if not self.videoFile:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(1.0)
        # otherwise, grab a reference to the video file
        else:
            vs = cv2.VideoCapture(self.videoFile)

        w = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

        reshape_ratio = np.array((w / float(self.reshapeW), h / float(self.reshapeH)))

        self.fps = vs.get(cv2.CAP_PROP_FPS)
        self.frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        self.spf = 1.0 / self.fps

        print(f"[INFO] fps={self.fps}, frames={self.frames}, spf={self.spf}(seconds per frame)")

        fps = FPS().start()

        while True:
            starttime = time.time()
            ret, frame = vs.read()
            reshaped_frame = cv2.resize(frame, (self.reshapeW, self.reshapeH))

            if not ret:
                break

            # construct a blob from the image and then perform a forward pass of
            # the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(reshaped_frame, 1.0, (self.reshapeW, self.reshapeH),
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

                cv2.polylines(frame, self.nmspolys, True, (0,0,255), 1)

            cv2.imshow(self.winName, frame)
            key = cv2.waitKey(1) & 0xFF
            fps.update()
            elapsed = time.time() - starttime
            if self.frameskip:
                vs.set(cv2.CAP_PROP_POS_FRAMES, vs.get(cv2.CAP_PROP_POS_FRAMES) + elapsed // self.spf)
            # if 'q' key pressed, break from the loop
            if key == ord("q"):
                break

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # if we are using a webcam, release the pointer
        if not self.videoFile:
            vs.stop()

        # otherwise, release the file pointer
        else:
            vs.release()

        # close all windows
        cv2.destroyAllWindows()


    def showAll(self):
        # do nothing
        pass

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
            return cv2.waitKey(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        videoFile = sys.argv[1]
    else:
        videoFile = "26393931_1641152753_mp4.mp4"
    sample = SceneTextVideo(videoFile)
    sample.detect()
