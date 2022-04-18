import cv2
import streamlit as st

st.title("Mediapipe Live Face/Pose Detection")
st.write("Real time face and pose detection using Google's mediapipe library: https://google.github.io/mediapipe/")
run = st.radio("Choose option:",('Face detection','Pose detection'))
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)


import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


while run == 'Pose detection':
    _, image = camera.read()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        image.flags.writeable = False
        #image = cv2.cvtColor(image)
        results = pose.process(image)
        mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        FRAME_WINDOW.image(image)
else:
    st.write('Stopped') 
    
while run == 'Face detection':
    _, image = camera.read()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
        image.flags.writeable = False
        #image = cv2.cvtColor(image)
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)#face detection
                #mp_drawing.draw_landmarks(
                #    image,
                #    results.pose_landmarks,
                #    mp_pose.POSE_CONNECTIONS,
                #    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        FRAME_WINDOW.image(image)
else:
    st.write('Stopped') 
    
    
