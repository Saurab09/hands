import cv2
import mediapipe as mp
import time


import numpy as np 
from mediapipe.framework.formats import landmark_pb2 
import time 
import win32api 
import pyautogui 


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



camera = cv2.VideoCapture(0)  

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while camera.isOpened():

        success, image = camera.read()

        start = time.time()
   

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape 


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and find hands
        results = hands.process(image)

        image.flags.writeable = True

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        '''if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)'''

        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)  
               # mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
 
        if results.multi_hand_landmarks != None: 
            for handLandmarks in results.multi_hand_landmarks: 
                for points in mp_hands.HandLandmark: 
                    normalizedLandmark = handLandmarks.landmark[points] 
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, image_width, image_height) 
                    points=str(points)
                    
                    if points == 'HandLandmark.INDEX_FINGER_TIP':
                        try: 
                                cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]),25, (0, 200, 0), 5) 
                                indexfingertip_x = pixelCoordinatesLandmark[0] 
                                indexfingertip_y = pixelCoordinatesLandmark[1] 
                               # print("jello")
                                win32api.SetCursorPos((indexfingertip_x*4, indexfingertip_y*5)) 
                                pyautogui.mouseDown(button='left') 
                        except:
                            pass   



        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
      #  print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('MediaPipe Hands', image)



        if cv2.waitKey(5) & 0xFF == ord('t'):
          break

camera.release()