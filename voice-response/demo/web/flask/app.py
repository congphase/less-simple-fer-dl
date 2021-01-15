from flask import Flask, render_template, Response
import cv2
import utils
from playsound import playsound
import random
import os
from mtcnn import MTCNN

app = Flask(__name__)

# Camera feed source
vid = cv2.VideoCapture(0)

# DEFINITIONS
N_FPS = vid.get(cv2.CAP_PROP_FPS)  
PREDICTION_TIMER = N_FPS*4
FONT = cv2.FONT_HERSHEY_SIMPLEX
N_FPS = vid.get(cv2.CAP_PROP_FPS)
print(f'fps: {N_FPS}')


prediction_timer = PREDICTION_TIMER
detector = MTCNN()

def gen_frames():  
    while True:
        success, frame = vid.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            global prediction_timer
            time_marker = int(prediction_timer/N_FPS)
            if(time_marker != (PREDICTION_TIMER/N_FPS)):
                cv2.putText(frame, str(time_marker), (10,450), FONT, 3, (0, 255, 0), 2, cv2.LINE_AA)

            # Decrease the timer
            prediction_timer -= 1

            if(prediction_timer == 0):
                print('Thực hiện predict!')
                detector.detect_faces()


                # tu cho label
                label = '0_angry'
                lines = utils.response_paths(label)
                random_response = random.choice(lines)
                playsound(random_response)
                prediction_timer = PREDICTION_TIMER

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

  

def predict(face):
    return label
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)