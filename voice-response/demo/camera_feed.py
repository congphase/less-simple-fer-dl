# DEFAULT WIDTH, HEIGHT = 640, 480

import cv2 
from playsound import playsound
import random
import os
import utils
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
# Check success
if not vid.isOpened():
    raise Exception("Could not open video device")


# DEFINITIONS
N_FPS = vid.get(cv2.CAP_PROP_FPS)  
PREDICTION_TIMER = N_FPS*4
FONT = cv2.FONT_HERSHEY_SIMPLEX

N_FPS = vid.get(cv2.CAP_PROP_FPS)
print(f'fps: {N_FPS}')

prediction_timer = PREDICTION_TIMER
while(True): 
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 

    # flip it horizontally
    frame = cv2.flip(frame, 1)

    time_marker = int(prediction_timer/N_FPS)
    if(time_marker != (PREDICTION_TIMER/N_FPS)):
        cv2.putText(frame, str(time_marker), (10,450), FONT, 3, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame 
    cv2.imshow('Video Feed', frame)

    # Decrease the timer
    prediction_timer -= 1

    if(prediction_timer == 0):
        print('Thực hiện predict!')

        # tu cho label
        label = '0_angry'
        lines = utils.response_paths(label)
        random_response = random.choice(lines)
        playsound(random_response)
        prediction_timer = PREDICTION_TIMER

        
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()