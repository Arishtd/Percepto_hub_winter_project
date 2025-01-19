import pickle
import cv2 as cv
import mediapipe as mp
import numpy as np

model_dict=pickle.load(open("./model.p", "rb"))
model=model_dict["model"]

cam=cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict={i:chr(65+i) for i in range(26)}

while True:
    data_aux=[]
    x_=[]
    y_=[]

    bool, frame=cam.read()
    if not bool:
        break

    H,W,Z=frame.shape
    frame_rgb=cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results=hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,0), circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0), circle_radius=2)
            )
    
        x1=int(min(x_)*W)-10
        y1=int(min(y_)*H)-10
        x2=int(min(x_)*W)+10
        y2=int(min(y_)*H)+10

        if len(data_aux)==42:
            prediction=model.predict([np.asanyarray(data_aux)])
            predicted_char=labels_dict[int(prediction[0])]

            cv.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
            cv.putText(frame, predicted_char, (x1,y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv.imshow("cam", frame)
    if cv.waitKey(20) & 0xFF == 27:
        break

cam.release()
cv.destroyAllWindows()