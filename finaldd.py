from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize the mixer for playing sounds
mixer.init()
# Load the background music
mixer.music.load(r"C:\Users\user\Desktop\New folder\music.wav")
# Load the ambulance sound and assign it to a specific channel
ambulance_sound = mixer.Sound(r"C:\Users\user\Desktop\New folder\ambulance.mp3")
ambulance_channel = mixer.Channel(1)  # Create a separate channel for the ambulance sound

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # Distance between vertical eye landmarks (1, 5)
    B = distance.euclidean(eye[2], eye[4])  # Distance between vertical eye landmarks (2, 4)
    C = distance.euclidean(eye[0], eye[3])  # Distance between horizontal eye landmarks (0, 3)
    ear = (A + B) / (2.0 * C)  # Calculate the EAR
    return ear

# Thresholds and parameters for drowsiness detection
thresh = 0.25  # EAR threshold for drowsiness
frame_check = 20  # Number of consecutive frames below threshold to trigger alert

# Initialize Haarcascade face detector and dlib's facial landmarks predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predict = dlib.shape_predictor(r"C:\Users\user\Desktop\New folder\shape_predictor_68_face_landmarks.dat")

# Get the indices for the left and right eyes in the facial landmarks array
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture from the default camera
cap = cv2.VideoCapture(0)
flag = 0  # Counter for consecutive frames with EAR below threshold
frame_count = 0  # Counter for total frames processed
line_y = None  # Y-coordinate for the line 1 cm below the forehead point
below_line_count = 0  # Counter for frames where the dot is below the line

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = imutils.resize(frame, width=850)  # Resize the frame to a width of 850 pixels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces in the grayscale frame
    
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predict(gray, rect)  # Predict facial landmarks for the detected face
        shape = face_utils.shape_to_np(shape)  # Convert the landmarks to a NumPy array
        
        leftEye = shape[lStart:lEnd]  # Get the landmarks for the left eye
        rightEye = shape[rStart:rEnd]  # Get the landmarks for the right eye
        leftEAR = eye_aspect_ratio(leftEye)  # Calculate the EAR for the left eye
        rightEAR = eye_aspect_ratio(rightEye)  # Calculate the EAR for the right eye
        ear = (leftEAR + rightEAR) / 2.0  # Average the EAR for both eyes
        
        # Draw contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1 , (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # New code to find the point above the nose and between the eyes
        nose_bridge_top = shape[27]  # Landmark index 27 is at the top of the nose bridge
        
        # Define a point above the nose bridge
        forehead_point = (nose_bridge_top[0], nose_bridge_top[1] - 20)
        
        # Draw a circle at this point
        cv2.circle(frame, forehead_point, 3, (255, 0, 0), -1)
        
        # Calculate the position for the line 1 cm below the forehead point
        pixels_for_1_cm = 38  # Approximate number of pixels for 1 cm
        
        if frame_count < 5:  # Set the line position based on initial frames
            line_y = forehead_point[1] + pixels_for_1_cm
        
        if line_y is not None:
            # Draw the fixed line
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)  # Line color green and thickness 2
        
        # Check if the blue dot goes below the green line
        if line_y is not None and forehead_point[1] > line_y:
            below_line_count += 1  # Increment counter if below the line
        else:
            below_line_count = 0  # Reset counter if the dot goes above the line

        # Play ambulance sound if the dot has been below the line for 20 frames
        if below_line_count >= 20 and not ambulance_channel.get_busy():
            mixer.music.stop()  # Stop any other sound playing
            ambulance_channel.play(ambulance_sound)
        
        if ear < thresh:
            flag += 1  # Increment flag if EAR is below threshold
            if flag >= frame_check:
                # Display alert on the frame
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    ambulance_channel.stop()  # Stop any other sound playing
                    mixer.music.play()
        else:
            flag = 0  # Reset flag if EAR is above threshold
    
    frame_count += 1  # Increment the frame counter
    cv2.imshow("Frame", frame)  # Display the frame with annotations
    key = cv2.waitKey(1) & 0xFF  # Capture key press
    if key == ord("q"):  # Exit loop if 'q' is pressed
        break

# Release the video capture and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()
