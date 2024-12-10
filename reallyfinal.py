from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

mixer.init()
music_channel_eyes = mixer.Channel(0)  # Channel for eye closing detection
music_channel_yawn = mixer.Channel(1)  # Channel for yawning detection

music_sound = mixer.Sound("C:\\Users\\user\\Desktop\\desktop everything in this folder\\drowsiness\\code\\music.wav")  # Load music.wav as a Sound object
                           
ambulance_sound = mixer.Sound("C:\\Users\\user\\Desktop\\desktop everything in this folder\\drowsiness\\code\\ambulance.mp3")
                               
ambulance_channel = mixer.Channel(2)  # Separate channel for the ambulance sound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])  # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

thresh = 0.25
frame_check = 20
yawn_thresh = 0.75
yawn_frame_check = 15

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("c:\\Users\\user\\Desktop\\desktop everything in this folder\\drowsiness\\code\\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag = 0
yawn_flag = 0
frame_count = 0
line_y = None
below_line_count = 0  # Counter for frames where the dot is below the line

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # New code to find the point above the nose and between the eyes
        nose_bridge_top = shape[27]
        
        # Define a point above the nose bridge
        forehead_point = (nose_bridge_top[0], nose_bridge_top[1] - 20)
        
        # Draw a circle at this point
        cv2.circle(frame, forehead_point, 3, (255, 0, 0), -1)
        
        # Calculate the position for the line 1 cm below the forehead point
        pixels_for_1_cm = 38
        
        if frame_count < 5:
            line_y = forehead_point[1] + pixels_for_1_cm
        
        if line_y is not None:
            # Draw the fixed line
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)  # Line color green and thickness 2
        
        # Check if the blue dot goes below the green line
        if forehead_point[1] > line_y:
            below_line_count += 1
        else:
            below_line_count = 0  # Reset counter if the dot goes above the line

        # Play ambulance sound if the dot has been below the line for 20 frames
        if below_line_count >= 20 and not ambulance_channel.get_busy():
            music_channel_eyes.stop()  # Stop any eye closing sound playing
            ambulance_channel.play(ambulance_sound)
        elif below_line_count < 20 and ambulance_channel.get_busy():
            ambulance_channel.stop()  # Stop the ambulance sound if the dot is above the line

        # Check for eye closure
        if ear < thresh and yawn_flag == 0:  # Check if not yawning
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not music_channel_eyes.get_busy():
                    music_channel_eyes.play(music_sound)
        else:
            flag = 0
            if music_channel_eyes.get_busy() and yawn_flag == 0:
                music_channel_eyes.stop()  # Stop the sound if eyes are open or yawning
        
        # Check for yawning
        if mar > yawn_thresh:
            yawn_flag += 1
            if yawn_flag >= yawn_frame_check:
                cv2.putText(frame, "**************YAWN ALERT!*************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "**************YAWN ALERT!*************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not music_channel_yawn.get_busy():
                    music_channel_yawn.play(music_sound)
        else:
            yawn_flag = 0
            if music_channel_yawn.get_busy():
                music_channel_yawn.stop()  # Stop the sound if yawning stops
    
    frame_count += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
