# -----------------------------------------------------------------------------
# Code created by: Mohammed Safwanul Islam @safwandotcom®
# Project: Computer Vision Data Science OPENPOSE 
# Date created: 15th November 2024
# Organization: N/A
# -----------------------------------------------------------------------------
# Description:
# This code captures live video from the webcam, applies pose estimation using MediaPipe, 
# and visualizes the detected body landmarks and connections in real time. The program runs continuously until the user presses the 'q' key to exit. 
# It demonstrates an application of computer vision for human pose tracking, which can be used in fields like fitness, gaming, and gesture recognition.
#### This code can detect only single person in detection ####
# -----------------------------------------------------------------------------
# License:
# This code belongs to @safwandotcom®.
# Code can be freely used for any purpose with proper attribution.
# -----------------------------------------------------------------------------
# Modules to install for this program to run using WINDOWS POWERSHELL
# pip install opencv-python
# pip install mediapipe

import cv2    #Imports the OpenCV library for computer vision tasks like image processing and video capture.
import mediapipe as mp  # Imports the MediaPipe library, providing pre-built machine learning pipelines like pose estimation.

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose  #Creates an instance of the mp.solutions.pose class, which handles the pose estimation process.
mp_drawing = mp.solutions.drawing_utils  #Provides utilities for drawing landmarks and connections on images.

# Initialize webcam capture
cap = cv2.VideoCapture(0) #Initializes video capture from the webcam using OpenCV. The argument 1 specifies the index of the webcam device (0 for the default camera).

# Customize drawing styles for landmarks and connections
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=int(2.5))  # Green for landmarks
connection_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Red for connections

#Creates a context manager for the mp.solutions.pose object. This ensures proper resource management (e.g., closing the pose model) after the loop exits.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
#Sets the minimum confidence threshold for detecting a person in the frame (default 0.5).    
#Sets the minimum confidence threshold for tracking a detected person across frames (default 0.5).
    while cap.isOpened(): #The main loop that continues as long as the webcam is successfully opened.
        success, image = cap.read() #Reads a frame from the webcam and stores it in the image variable. success indicates whether the frame was read successfully.
        if not success: #If success is False, an error message is printed and the loop continues to the next frame.
            print("Ignoring empty camera frame.")
            continue

        #Converts the image from BGR (OpenCV's default color format) to RGB format, which MediaPipe expects.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        #This line improves performance by marking the image as not writeable. MediaPipe can then modify the image directly, avoiding unnecessary copying.
        image_rgb.flags.writeable = False

        #Processes the RGB image using the MediaPipe Pose model to detect body keypoints and generate a results object containing the pose information.
        results = pose.process(image_rgb)

        # Mark the image as writeable again to draw on it
        image_rgb.flags.writeable = True

        # Convert back to BGR for displaying with OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # If landmarks are detected, draw them with the customized style
        if results.pose_landmarks: #This condition checks if the results object contains valid pose landmarks. If no landmarks are detected, the code inside the if block won't execute.
            mp_drawing.draw_landmarks( #This line calls the draw_landmarks function from the mp_drawing module to visualize the pose landmarks and connections. 
                image_bgr, #The image on which the landmarks and connections will be drawn.
                results.pose_landmarks, #The detected pose landmarks, which are represented as a list of normalized coordinates.
                mp_pose.POSE_CONNECTIONS,  # Pose connection lines (e.g., between keypoints)
                landmark_style,            # Style for landmarks (e.g., color, thickness)
                connection_style           # Style for connections (e.g., color, thickness)
            )

        # Display the resulting image
        cv2.imshow('Open Pose by Safwanul', image_bgr)

        # Exit on pressing the 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
# This method releases the resources associated with the video capture device. 
# This is crucial to prevent resource leaks and ensure proper system operation. 
# By releasing the capture device, other applications can access it without interference.
cap.release() 

cv2.destroyAllWindows() 
#This above function closes all OpenCV windows that are currently open. 
# #This is important to clean up the environment and avoid potential issues when running multiple scripts or applications that use OpenCV.