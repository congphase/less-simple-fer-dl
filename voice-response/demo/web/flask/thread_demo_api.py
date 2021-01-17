import cv2
import argparse
import os
from CountsPerSec import CountsPerSec
#from VideoGet import VideoGet
from VideoShow import VideoShow

import utils
from playsound import playsound
import random
from time import time
from keras.models import load_model
from keras_vggface.utils import preprocess_input
import numpy as np

from flask import Flask, render_template, Response
from threading import Thread
import requests

app = Flask(__name__)


# DEFINITIONS
N_FPS = 30
PREDICTION_TIMER = N_FPS*6
FONT = cv2.FONT_HERSHEY_SIMPLEX
print(f'fps: {N_FPS}')


prediction_timer = PREDICTION_TIMER
#detector = MTCNN()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model(r'C:\Users\phalc\Documents\Dao-Tao\Course-Deep-Learning\Project\Cuoi-Ky\dl-end-term\voice-response\demo\models\vggface1_ferplus.h5')

label_converter = {
    0: 'Neutral', 
    1: 'Happiness', 
    2: 'Surprise', 
    3: 'Sadness', 
    4: 'Anger', 
    5: 'Disgust', 
    6: 'Fear', 
    7: 'Contempt'
}


label_converter_2 = {
    'Neutral': '6_neutral',
    'Happiness': '3_happy', 
    'Surprise': '5_surprise',
    'Sadness': '4_sad',
    'Anger': '0_angry',
    'Disgust': '1_disgust', 
    'Fear': '2_fear', 
    'Contempt': '1_disgust'
}


def predict(face_img):
    face_img = face_img.astype('float32')
    # resize image to the target size 
    x = cv2.resize(face_img, (224,224), interpolation=cv2.INTER_CUBIC)
    #x = cv2.resize(face_img, (224,224))
    x = preprocess_input(x, version=2) # or version=2 for VGGFace2 ResNet50  
    x = np.expand_dims(x, axis=0)
    label = model.predict(x)
    return label


def talk(audio=''):
    playsound(audio)


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def stop(self):
        self.stopped = True        

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                start_ = time()
                self.frame = cv2.flip(self.frame, 1)
                frame_ = self.frame
                gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
                detection = face_cascade.detectMultiScale(gray, 1.3, 5)

                if(len(detection)==1):
                    # draw bounding boxes
                    x=detection[0][0]
                    y=detection[0][1]
                    w=detection[0][2]
                    h=detection[0][3]
                    face_img = frame_[y:y+h, x:x+w]         
                    frame_ = cv2.rectangle(frame_, (x, y), (x+w, y+h), (255,255,0), 1)

                #elif(len(detection)>1): # not implemented

                global prediction_timer
                time_marker = int(prediction_timer/N_FPS)
                if(time_marker != (PREDICTION_TIMER/N_FPS)):
                    cv2.putText(frame_, str(time_marker), (10,450), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Decrease the timer
                prediction_timer -= 1

                if(prediction_timer == 0 and len(detection) == 1):
                    print('Thực hiện predict!')
                    start = time()

                    label = predict(face_img)
                    end = time()
                    print(f'DEBUG: Prediction time: {(end-start):.3f}s')
                    print(f'DEBUG: Overall time: {time()-start_}s')

                    label = np.argmax(label, axis=-1)[0]
                    label = label_converter[label]
                    label = label_converter_2[label]
                    print(f'DEBUG: lbbbbbbbbbbbbb: {label}')

                    frame_ = cv2.rectangle(frame_, (x, y+h), (x+w, y+h+30), (255,255,0), -1)
                    #cv2.putText(frame_, text=label[2:], org=(x, y+h+25), fontFace=FONT, fontScale=1, \
                    #    color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    

                    # prepare the coresponding response
                    lines = utils.response_paths(label)
                    random_response = random.choice(lines)
                    
                    # spawn a new thread for talking only
                    audio_thread = Thread(target=talk, args=(random_response,))
                    audio_thread.start()
                    #begin_ = time()
                    #playsound(random_response)
                    #print(f'DEBUG: speak time: {time()-begin_}s')

                if(prediction_timer==0):
                    prediction_timer = PREDICTION_TIMER





    

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def noThreading(source=0):
    """Grab and show video frames without multithreading."""

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def threadVideoShow(source=0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default=0,
        help="Path to video file or integer representing webcam index"
            + " (default 0).")
    ap.add_argument("--thread", "-t", default="none",
        help="Threading mode: get (video read in its own thread),"
            + " show (video show in its own thread), both"
            + " (video read and video show in their own threads),"
            + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    # If source is a string consisting only of integers, check that it doesn't
    # refer to a file. If it doesn't, assume it's an integer camera ID and
    # convert to int.
    if (
        isinstance(args["source"], str)
        and args["source"].isdigit()
        and not os.path.isfile(args["source"])
    ):
        args["source"] = int(args["source"])

    if args["thread"] == "both":
        threadBoth(args["source"])
    elif args["thread"] == "get":
        threadVideoGet(args["source"])
    elif args["thread"] == "show":
        threadVideoShow(args["source"])
    else:
        noThreading(args["source"])

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(threadVideoGet(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)