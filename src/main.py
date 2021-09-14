import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from data_preprocess import X_train
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
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = trained_model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
# while True:
#     ''' Find haar cascade to draw bounding box around face'''
#     ret, frame = cap.read()
#     if not ret:
#         break

#     ''' Detect faces in the image'''
#     results = detector.detect_faces(frame)

#     ''' Extract the bounding box from the first face '''
#     if len(results) == 1:  # if face detected = 1, else = 0
#         try:
#             x1, y1, width, height = results[0]['box']
#             x2, y2 = x1 + width, y1 + height

#             ''' Extract face '''
#             face = frame[y1:y2, x1:x2]

#             ''' Draw bounding box '''
#             cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (255, 0, 0), 2)
            
#             ''' Resize pixels to the model size'''
#             cropped_img = cv2.resize(face, (48, 48))
#             cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) 
#             cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
#             cropped_img_float = cropped_img_expanded.astype(float)

#             ''' Model prediction'''
#             prediction = trained_model.predict(cropped_img_float)
#             print(prediction)
#             maxindex = int(np.argmax(prediction))

#             cv2.putText(frame, 'HHHHH', (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (219, 68, 55), 2, cv2.LINE_AA)
#             # cv2.putText(frame, emotion_dict[maxindex], (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#         except:
#             pass
     

#     cv2.imshow('Video',frame)
#     try:
#         cv2.imshow("frame", cropped_img)
#     except:
#         cv2.imshow("frame", black)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()