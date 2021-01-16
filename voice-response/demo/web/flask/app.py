from flask import Flask, render_template, Response
import cv2
import utils
from playsound import playsound
import random
import os
from time import time
from keras.models import load_model
from keras_vggface.utils import preprocess_input
import numpy as np

app = Flask(__name__)

# Camera feed source
vid = cv2.VideoCapture(0)

# DEFINITIONS
N_FPS = vid.get(cv2.CAP_PROP_FPS)  
PREDICTION_TIMER = N_FPS*3
FONT = cv2.FONT_HERSHEY_SIMPLEX
print(f'fps: {N_FPS}')


prediction_timer = PREDICTION_TIMER
#detector = MTCNN()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model(r'C:\Users\phalc\Documents\Dao-Tao\Course-Deep-Learning\Project\Cuoi-Ky\dl-end-term\voice-response\demo\models\VGGFACE2_V1_RandomOverSampling.h5')

label_converter = {
    0: '0_angry',
    1: '1_disgust',
    2: '2_fear',
    3: '3_happy',
    4: '4_sad',
    5: '5_surprise',
    6: '6_neutral'
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

def gen_frames():  
    while True:
        success, frame = vid.read()  # read the camera frame
        start_ = time()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = face_cascade.detectMultiScale(gray, 1.3, 5)

            if(len(detection)==1):
                # draw bounding boxes
                x=detection[0][0]
                y=detection[0][1]
                w=detection[0][2]
                h=detection[0][3]
                face_img = frame[y:y+h, x:x+w]         
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 1)

            #elif(len(detection)>1): # not implemented

            global prediction_timer
            time_marker = int(prediction_timer/N_FPS)
            if(time_marker != (PREDICTION_TIMER/N_FPS)):
                cv2.putText(frame, str(time_marker), (10,450), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
                #cv2.imwrite(f'predict_{label}.jpg', face_img)

                frame = cv2.rectangle(frame, (x, y+h), (x+w, y+h+30), (255,255,0), -1)
                cv2.putText(frame, text=label[2:], org=(x, y+h+25), fontFace=FONT, fontScale=1, \
                    color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                #cv2.imwrite(f'frame_{x}.jpg', frame)

                # prepare the coresponding response
                lines = utils.response_paths(label)
                random_response = random.choice(lines)
                playsound(random_response)

            if(prediction_timer==0):
                prediction_timer = PREDICTION_TIMER

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

  
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