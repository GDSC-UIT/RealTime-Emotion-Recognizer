import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
black = np.zeros((96,96))

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break

	# detect faces in the image
    results = detector.detect_faces(frame)
	
    # extract the bounding box from the first face
    if len(results) == 1:  # if face detected = 1, else = 0
        try:
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height

        	# extract face
            face = frame[y1:y2, x1:x2]

            #Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (255, 0, 0), 2)
            
            # resize pixels to the model size
            cropped_img = cv2.resize(face, (48, 48)) 
            cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
            cropped_img_float = cropped_img_expanded.astype(float)
            # prediction = trained_model.predict(cropped_img_float)
            # print(prediction)
            # maxindex = int(np.argmax(prediction))

            cv2.putText(frame, 'HHHHH', (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (219, 68, 55), 2, cv2.LINE_AA)
            # cv2.putText(frame, emotion_dict[maxindex], (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass
        
    cv2.imshow('Video',frame)
    try:
        cv2.imshow("frame", cropped_img)
    except:
        cv2.imshow("frame", black)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()