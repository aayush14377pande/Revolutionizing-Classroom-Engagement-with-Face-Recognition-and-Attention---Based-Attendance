import cv2
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import time

# Load the Caffe model files with single line paths
prototxt_path = '/Users/aayushpande/MULTIPLE_FACES/Models/deploy.prototxt'
model_path = '/Users/aayushpande/MULTIPLE_FACES/Models/res10_300x300_ssd_iter_140000.caffemodel'

# Verify the paths
print("Prototxt Path:", prototxt_path)
print("Model Path:", model_path)

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load face and names data
data_folder = '/Users/aayushpande/MULTIPLE_FACES/'
faces_file = os.path.join(data_folder, 'faces_data.pkl')
names_file = os.path.join(data_folder, 'names.pkl')

# Load the faces and names data
with open(faces_file, 'rb') as f:
    faces_data = pickle.load(f)

with open(names_file, 'rb') as f:
    names = pickle.load(f)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(faces_data, names)

# Create the directory for attendance files if it doesn't exist
attendance_folder = os.path.join(data_folder, 'ATTENDANCE SHEETS')
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Get the current date and time to create a unique attendance file name
now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
attendance_file = os.path.join(attendance_folder, f'attendance_{date_time_str}.csv')

# Initialize attendance DataFrame with columns
attendance = pd.DataFrame(columns=["Name", "Start Time", "End Time", "Total Time", "Alert Time", "Alertness", "Attendance"])

# Dictionary to track marked faces and their details
marked_faces = {}
video = cv2.VideoCapture(0)  # Access the webcam

# Record the time when the camera starts
camera_start_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Flip the camera feed horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Prepare the frame for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Perform face detection
    net.setInput(blob)
    detections = net.forward()

    current_faces = set()  # Track currently detected faces

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Set a threshold for detection confidence
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            face = frame[y:y1, x:x1]

            # Get prediction from the KNN model
            resized_img = cv2.resize(face, (50, 50)).reshape(1, -1)
            label = knn.predict(resized_img)[0]

            # Mark current face
            current_faces.add(label)

            current_time = time.time()
            
            if label not in marked_faces:
                # New face detected, set start time as the camera start time
                marked_faces[label] = {
                    "Start Time": camera_start_time,  # All faces have the same start time
                    "End Time": None,
                    "Total Time": 0,
                    "Alert Time": 0,
                    "Last Detection Time": current_time,
                    "Detected": True
                }
                print(f"[{datetime.now()}] Face detected: {label}")
            else:
                # Update alert time for continuous detection
                if marked_faces[label]["Detected"]:
                    # Calculate time since last detection
                    elapsed_time = current_time - marked_faces[label]["Last Detection Time"]
                    marked_faces[label]["Alert Time"] += elapsed_time  # Update alert time
                    marked_faces[label]["Last Detection Time"] = current_time  # Update last detection time
                    print(f"[{datetime.now()}] Face still detected: {label} | Alert Time: {marked_faces[label]['Alert Time']}")
                else:
                    # If the face was previously lost, reset detection tracking
                    marked_faces[label]["Detected"] = True
                    marked_faces[label]["Last Detection Time"] = current_time  # Update last detection time
                    print(f"[{datetime.now()}] Face re-detected: {label}")

            # Draw rectangle around detected faces
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

    # Update times for faces that are not currently detected
    for name in list(marked_faces.keys()):
        if name not in current_faces:
            if marked_faces[name]["Detected"]:
                # If the face was previously detected, finalize its detection
                end_time = time.time()
                marked_faces[name]["End Time"] = end_time
                marked_faces[name]["Total Time"] = round(end_time - marked_faces[name]["Start Time"], 2)  # Round Total Time to 2 decimal places
                marked_faces[name]["Detected"] = False  # Mark as no longer detected
                print(f"[{datetime.now()}] Face lost: {name}")

    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Process final entries for attendance
for name, details in marked_faces.items():
    end_time = time.time()
    details["End Time"] = end_time
    details["Total Time"] = round(end_time - details["Start Time"], 2)  # Round Total Time to 2 decimal places

    # If the face was still detected, finalize the alert time
    if details["Detected"]:
        # Calculate remaining time since the last detection until end time
        remaining_alert_time = end_time - details["Last Detection Time"]
        details["Alert Time"] += remaining_alert_time

    # Round Alert Time to 2 decimal places
    details["Alert Time"] = round(details["Alert Time"], 2)

    print(f"Final Alert Time for {name}: {details['Alert Time']} seconds")

    # Calculate alertness percentage and attendance status
    alertness_percentage = round(details['Alert Time'] / details['Total Time'] * 100, 2)
    attendance_status = "Present" if alertness_percentage > 75 else "Absent"

    # Prepare the entry for the attendance DataFrame
    new_entry = pd.DataFrame({
        "Name": [name],
        "Start Time": [datetime.fromtimestamp(details["Start Time"]).strftime('%Y-%m-%d %H:%M:%S')],
        "End Time": [datetime.fromtimestamp(details["End Time"]).strftime('%Y-%m-%d %H:%M:%S')],
        "Total Time": [details["Total Time"]],
        "Alert Time": [details["Alert Time"]],
        "Alertness": [f"{alertness_percentage}%"],  # Add '%' sign
        "Attendance": [attendance_status]
    })

    # Add the entry to attendance DataFrame
    attendance = pd.concat([attendance, new_entry], ignore_index=True)

# Save attendance data to the CSV file
attendance.to_csv(attendance_file, index=False)
print(f"Attendance saved to {attendance_file}")

# Release resources
video.release()
cv2.destroyAllWindows()
