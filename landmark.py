import os
import pickle
import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

data_dir = "./data"
annotated_data = "./annotated_data"
data = []
labels = []

if not os.path.exists(annotated_data):
    os.makedirs(annotated_data)

for dir in os.listdir(data_dir):
    annotated_data_dir = os.path.join(annotated_data, dir)
    if not os.path.exists(annotated_data_dir):
        os.makedirs(annotated_data_dir)

for dir_ in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir_)
    if os.path.isdir(dir_path):  
        label_dir = os.path.join(annotated_data, dir_)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        for img_path in os.listdir(dir_path):
            img = cv.imread(os.path.join(dir_path, img_path))

            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)
                    
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    annotated_img_path = os.path.join(label_dir, img_path)
                    base_name, ext = os.path.splitext(img_path)
                    count = 1
                    while os.path.exists(annotated_img_path):
                        annotated_img_path = os.path.join(label_dir, f"{base_name}_{count}{ext}")
                        count += 1
                    
                    cv.imwrite(annotated_img_path, img)

                data.append(data_aux)
                labels.append(dir_)

f=open("data.pickle", "wb")
pickle.dump({"data" : data, "labels" : labels}, f)
f.close()