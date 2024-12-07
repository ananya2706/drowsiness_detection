import cv2
from scipy.spatial import distance as dist
import winsound

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    if len(eye) != 4:
        return 0.0
    
    A = dist.euclidean(eye[1], eye[2])
    B = dist.euclidean(eye[1], eye[3])
    C = dist.euclidean(eye[2], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define constants for drowsiness detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
COUNTER = 0
ALARM_ON = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(eyes) == 0:
            winsound.Beep(1000, 200)  # Sound alert
            
        for (ex, ey, ew, eh) in eyes:
            # Extract eye landmarks
            eye_landmarks = [
                (ex, ey + eh // 2),
                (ex + ew // 3, ey + eh // 2),
                (ex + 2 * ew // 3, ey + eh // 2),
                (ex + ew, ey + eh // 2)
            ]
            # Calculate eye aspect ratio (EAR)
            ear = eye_aspect_ratio(eye_landmarks)
            
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # Sound alarm
                        winsound.Beep(1000, 200)
            else:
                COUNTER = 0
                ALARM_ON = False

    cv2.imshow('Drowsiness Detection', frame)
    
    # Exit the loop when 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # 'Esc' key
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
