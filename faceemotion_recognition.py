from  facial_emotion_recognition import EmotionRecognition

import cv2

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(1)

while True:
    success, frame = cam.read()
    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow('frame',frame)
    key = cv2.waitkey(1)
    if key == 'q':
        break

cam.release()
cv2.destroyAllWindows()

 
