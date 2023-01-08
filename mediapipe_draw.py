import cv2 as cv
import mediapipe as mp
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout


def image_draw(file):
    filestr = file.read()
    file_bytes = np.fromstring(filestr, np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    # mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法   
    with mp_hands.Hands(
        static_image_mode=True, 
        model_complexity=1, 
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
        max_num_hands=1) as hands:      
        while True:
            if not img.any():
                print("Cannot receive frame")
                break
             # 偵測手掌
            results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))   
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                    wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                    wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z
                    thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x 
                    thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y 
                    thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z
                    thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x 
                    thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
                    thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z
                    thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x 
                    thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y 
                    thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z 
                    thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x 
                    thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y 
                    thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z 
                    index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x 
                    index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y 
                    index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z 
                    index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                    index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y 
                    index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z 
                    index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
                    index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                    index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z 
                    index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x 
                    index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y 
                    index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z 
                    middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x 
                    middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y 
                    middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z 
                    middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x 
                    middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y 
                    middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z 
                    middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x 
                    middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y 
                    middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z 
                    middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                    middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                    middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                    ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
                    ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
                    ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z
                    ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x 
                    ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y 
                    ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z
                    ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x 
                    ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y 
                    ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z 
                    ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x 
                    ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y 
                    ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
                    pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x 
                    pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y 
                    pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z 
                    pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x 
                    pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y 
                    pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z 
                    pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x 
                    pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y 
                    pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z 
                    pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x 
                    pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y 
                    pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z 

                    vecter = [
                        thumb_CmcX - wristX,
                        thumb_CmcY - wristY,
                        thumb_CmcZ - wristZ,
                        thumb_McpX - thumb_CmcX,
                        thumb_McpY - thumb_CmcY,
                        thumb_McpZ - thumb_CmcZ,
                        thumb_IpX - thumb_McpX,
                        thumb_IpY - thumb_McpY,
                        thumb_IpZ - thumb_McpZ,
                        thumb_TipX - thumb_IpX,
                        thumb_TipY - thumb_IpY,
                        thumb_TipZ - thumb_IpZ,
                        # Index Finger
                        index_McpX - wristX,
                        index_McpY - wristY,
                        index_McpZ - wristZ,
                        index_PipX - index_McpX,
                        index_PipY - index_McpY,
                        index_PipZ - index_McpZ,
                        index_DipX - index_PipX,
                        index_DipY - index_PipY,
                        index_DipZ - index_PipZ,
                        index_TipX - index_DipX,
                        index_TipY - index_DipY,
                        index_TipZ - index_DipZ,
                        # Middle Finger
                        middle_McpX - index_McpX,
                        middle_McpY - index_McpY,
                        middle_McpZ - index_McpZ,
                        middle_PipX - middle_McpX,
                        middle_PipX - middle_McpY,
                        middle_PipX - middle_McpZ,
                        middle_DipX - middle_PipX,
                        middle_DipY - middle_PipY,
                        middle_DipZ - middle_PipZ,
                        middle_TipX - middle_DipX,
                        middle_TipY - middle_DipY,
                        middle_TipZ - middle_DipZ,
                        # Ring Finger
                        ring_McpX - middle_McpX,
                        ring_McpY - middle_McpY,
                        ring_McpZ - middle_McpZ,
                        ring_PipX - ring_McpX,
                        ring_PipY - ring_McpY,
                        ring_PipZ - ring_McpZ,
                        ring_DipX - ring_PipX,
                        ring_DipY - ring_PipY,
                        ring_DipZ - ring_PipZ,
                        ring_TipX - ring_DipX,
                        ring_TipY - ring_DipY,
                        ring_TipZ - ring_DipZ,
                        # Pinky Finger
                        pinky_McpX - wristX,
                        pinky_McpY - wristY,
                        pinky_McpZ - wristZ,
                        pinky_PipX - pinky_McpX,
                        pinky_PipY - pinky_McpY,
                        pinky_PipZ - pinky_McpZ,
                        pinky_DipX - pinky_PipX,
                        pinky_DipY - pinky_PipY,
                        pinky_DipZ - pinky_PipZ,
                        pinky_TipX - pinky_DipX,
                        pinky_TipY - pinky_DipY,
                        pinky_TipZ - pinky_DipZ
                    ]
                    # 將節點和骨架繪製到影像中
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
                        # mp_drawing_styles.get_default_hand_landmarks_style(),
                        # mp_drawing_styles.get_default_hand_connections_style())
                # 讀取模型以預測
                model = keras.models.load_model("./static/model2_b256k7_fps.h5", compile=False)
                model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                vecter = np.reshape(vecter, (-1, 60, 1))
                prediction = model.predict(vecter)
                return img, np.argmax(prediction)
    cv.destroyAllWindows()

def video_draw(file, filename):
    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法
    file_bytes = file.read()
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(file_bytes)
        cap = cv.VideoCapture(temp.name)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')          # 設定影片的格式為 MJPG"
    out1 = cv.VideoWriter(F'./static/storage/{filename}', fourcc, 30.0, (480,  480))  # 產生空的影片"
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1) as hands:
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))     # 偵測手掌
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 將節點和骨架繪製到影像中
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            out1.write(img)    # 寫入影片
    cap.release()
    out1.release()
    cv.destroyAllWindows()