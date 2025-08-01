import cv2
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import time

# Load the Haarcascade XML file
haar_path = '/Users/aayushpande/MULTIPLE_FACES/haarcascade_frontalface_default.xml'
if not os.path.exists(haar_path):
    raise FileNotFoundError(f"Haarcascade file not found at {haar_path}")
facedetect = cv2.CascadeClassifier(haar_path)

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
lost_faces = {}  # Temporary storage for lost faces
video = cv2.VideoCapture(0)  # Access the webcam

# Record the time when the camera starts
camera_start_time = time.time()

# Additional variables for tracking detection
min_detection_time = 2  # Minimum seconds a face must be detected to be considered valid
detection_times = {}  # Track detection times for faces
tracking_buffer_time = 5  # Time in seconds to keep track of lost faces
min_visible_time = 0.5  # Minimum visible time in seconds for a face to be considered valid

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Flip the camera feed horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast using histogram equalization
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))

    current_faces = set()  # Track currently detected faces
    transient_faces = {}  # Track transient faces for ignoring short detections

    for (x, y, w, h) in faces:
        if w * h > 2000:  # Area threshold
            face = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(face, (50, 50)).reshape(1, -1)
            label = knn.predict(resized_img)[0]
            confidence_score = knn.predict_proba(resized_img).max()

            # Check confidence threshold
            if confidence_score > 0.8:  # Confidence threshold
                current_faces.add(label)
                current_time = time.time()

                # Track face visibility time
                if label not in transient_faces:
                    transient_faces[label] = current_time
                else:
                    visible_time = current_time - transient_faces[label]
                    if visible_time >= min_visible_time and label not in marked_faces:
                        # Only add to marked faces if it has been visible long enough
                        marked_faces[label] = {
                            "Start Time": camera_start_time,
                            "End Time": None,
                            "Total Time": 0,
                            "Alert Time": 0,
                            "Last Detection Time": current_time,
                            "Detected": True
                        }
                        detection_times[label] = 0
                        lost_faces.pop(label, None)  # Remove from lost faces if detected
                        print(f"[{datetime.now()}] Face detected: {label}")

    # Update times for faces that are not currently detected
    for name in list(marked_faces.keys()):
        if name not in current_faces:
            # If face is lost, move it to lost_faces
            if marked_faces[name]["Detected"]:
                lost_faces[name] = marked_faces[name]  # Keep track of lost face details
                lost_faces[name]["Lost Time"] = time.time()  # Record when it was lost
                marked_faces[name]["Detected"] = False  # Mark as no longer detected
                print(f"[{datetime.now()}] Face lost: {name}")

            # Check if we should remove the lost face after the buffer time
            if name in lost_faces:
                if time.time() - lost_faces[name]["Lost Time"] > tracking_buffer_time:
                    del lost_faces[name]

    # Re-establish identities for lost faces that reappear
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(face, (50, 50)).reshape(1, -1)
        label = knn.predict(resized_img)[0]
        confidence_score = knn.predict_proba(resized_img).max()

        if label in lost_faces and confidence_score > 0.8:
            # Found a lost face
            current_faces.add(label)
            print(f"[{datetime.now()}] Lost face detected again: {label}")
            # Move back to marked faces
            marked_faces[label] = lost_faces[label]
            marked_faces[label]["Detected"] = True
            del lost_faces[label]  # Remove from lost faces

    # Display the video feed with face detection
    for label in current_faces:
        for (x, y, w, h) in faces:
            if label in marked_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Process final entries for attendance
for name, details in marked_faces.items():
    end_time = time.time()
    details["End Time"] = end_time
    details["Total Time"] = round(end_time - details["Start Time"], 2)

    if details["Detected"]:
        remaining_alert_time = end_time - details["Last Detection Time"]
        details["Alert Time"] += remaining_alert_time

    details["Alert Time"] = round(details["Alert Time"], 2)

    print(f"Final Alert Time for {name}: {details['Alert Time']} seconds")

    alertness_percentage = round(details['Alert Time'] / details['Total Time'] * 100, 2)
    attendance_status = "Present" if alertness_percentage > 75 else "Absent"

    new_entry = pd.DataFrame({
        "Name": [name],
        "Start Time": [datetime.fromtimestamp(details["Start Time"]).strftime('%Y-%m-%d %H:%M:%S')],
        "End Time": [datetime.fromtimestamp(details["End Time"]).strftime('%Y-%m-%d %H:%M:%S')],
        "Total Time": [details["Total Time"]],
        "Alert Time": [details["Alert Time"]],
        "Alertness": [f"{alertness_percentage}%"],
        "Attendance": [attendance_status]
    })

    attendance = pd.concat([attendance, new_entry], ignore_index=True)

# Save attendance data to the CSV file
attendance.to_csv(attendance_file, index=False)
print(f"Attendance saved to {attendance_file}")

# Release resources
video.release()
cv2.destroyAllWindows()
