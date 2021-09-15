import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
# from data_preprocess import X_train
from tensorflow.keras.preprocessing import image

#load model
trained_model = model_from_json(open("model/vgg-face-model.json", "r").read())
#load weights
trained_model.load_weights('model/vgg-face.h5')

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 30)
black = np.zeros((96,96))

emotions = ('happy', 'sad', 'neutral')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:

    ret,test_img = cap.read()# captures frame and returns boolean value and captured image

    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0), thickness=4)

        roi_gray = gray_img[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48,48))
        img_pixels = image.img_to_array(roi_gray)
        cv2.imshow('Gray img', roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = trained_model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (219,68,55), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Predicted image', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows