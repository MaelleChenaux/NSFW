from clarifai.rest import ClarifaiApp
from clarifai.rest import Workflow
from clarifai.rest import Image as ClImage
import numpy as np
import cv2
import threading
import math
import os
import time

app = ClarifaiApp(api_key='c09a76941d80407a969130873e5de42d')
model = app.models.get('nsfw-v1.0')
#model = app.models.get("moderation")
value = 0
targetValue = 0

result_received = True

# Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,40)
fontScale              = 1
fontColor              = (0, 0, 0)
lineType               = 2

frame = None

def send_to_ai_thread():
    global targetValue
    global result_received
    global photoFrame
    global frame

    while True:
        try:
            image = ClImage(file_obj=open('/Users/maellechenaux/ECAL/Workshop/code/save.png', 'rb'))
            data = model.predict([image])

            amountTags = len(data["outputs"][0]["data"]["concepts"])
            print("Found tags:" + str(amountTags))

            targetValue = 0
            concepts = data["outputs"][0]["data"]["concepts"]
            for concept in concepts:
                tagName = concept["name"]
                v = concept["value"]
                if tagName == 'nsfw':
                    targetValue = v
                    break

            print('NOT SAFE FOR WORK ---> '  + str(targetValue) + ' <--- NOT SAFE FOR WORK')
            os.rename('save.png', 'results/res_' + str(targetValue) + '.png')
        except Exception as e:
            print('Could not open save.png')

        if frame is None:
            continue

        framecp = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        cv2.imwrite('save.png', framecp)
        photoFrame = framecp

cap = cv2.VideoCapture(0)

photoFrame = None
threading.Thread(target=send_to_ai_thread).start()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    d = math.fabs(targetValue - value)
    direction = np.sign(targetValue - value)

    value = value + 0.3 * d * direction

    mapping = int(value * 200)

    if mapping % 2 == 0:
        mapping = mapping + 1

    if value < 0.1:
        mapping = 1

    #print("target value "+ str(targetValue) + " value " + str(value))
    dst = cv2.blur(frame, (mapping ,mapping))
    #dst = cv2.blur(frame, (0,0),cv2.BORDER_DEFAULT)

    cv2.imshow('Are you safe for work ?', dst)
    if photoFrame is not None:
        cv2.putText(photoFrame, str(targetValue), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow('Analysed', photoFrame)


    wk = cv2.waitKey(1)
    if wk & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
