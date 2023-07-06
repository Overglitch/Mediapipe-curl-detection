import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Curl counter variables
counter = 0
stage = None
previous_stage = None

## Setup mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load video file
cap = cv2.VideoCapture('curl.mp4')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic
            if angle > 160:
                stage = "abajo"
            if angle < 30 and stage == 'abajo' and previous_stage != 'arriba':
                stage = "arriba"
                counter += 1
                print(counter)

            previous_stage = stage

        except:
            pass

        # Render curl counter
        background_width = 250
        background_height = 100
        text_offset_x = 20
        text_offset_y = 100

        # Rep data
        cv2.putText(image, 'REPETICIONES:', (image.shape[1] - background_width + text_offset_x, text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (image.shape[1] - background_width + text_offset_x, text_offset_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'ESTADO:', (image.shape[1] - background_width + text_offset_x, text_offset_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (image.shape[1] - background_width + text_offset_x, text_offset_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Display image
        cv2.imshow('ACTIVIDAD MEDIAPIPE', image)

        # Break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#REFERENCIAS
"""
https://github.com/nicknochnack/MediaPipePoseEstimation
https://youtu.be/0zhvUV1bAVQ
código de clase de la semana 13
"""