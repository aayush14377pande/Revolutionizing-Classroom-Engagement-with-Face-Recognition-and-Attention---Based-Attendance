import cv2  # type: ignore
import pickle
import numpy as np  # type: ignore
import os
import time

# Access the webcam
video = cv2.VideoCapture(0)

# Path to the DNN model and Haarcascade files
dnn_prototxt = '/Users/aayushpande/MULTIPLE_FACES/Models/deploy.prototxt'
dnn_model = '/Users/aayushpande/MULTIPLE_FACES/Models/res10_300x300_ssd_iter_140000.caffemodel'

if not os.path.exists(dnn_prototxt) or not os.path.exists(dnn_model):
    raise FileNotFoundError(f"Model files not found in the specified paths.")

# Load the DNN model for face detection
net = cv2.dnn.readNetFromCaffe(dnn_prototxt, dnn_model)

faces_data = []
name = input("Enter Your Name: ")

max_duration = 10  # Total session duration in seconds
start_time = time.time()  # Start timer

# Store detected faces
face_rectangles = []

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video")
        break

    # Invert the frame
    frame = cv2.flip(frame, 1)

    # Create a blob from the frame for DNN face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    # Clear previous detections for fresh processing
    face_rectangles.clear()

    (h, w) = frame.shape[:2]
    boxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Adjusted confidence threshold to detect faces at different angles
        if confidence > 0.6:  # Reduced threshold slightly
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            boxes.append([x, y, x1 - x, y1 - y])
            confidences.append(float(confidence))

    # Apply Non-Maximum Suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

    # Check if indices is not empty
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]

            # Only process sufficiently large faces
            if w > 80 and h > 80:
                face_rectangles.append((x, y, w, h))
                face_region = frame[y:y + h, x:x + w]

                # Resize the face region for storage, keeping 3 channels (color)
                resized_img = cv2.resize(face_region, (50, 50), interpolation=cv2.INTER_AREA)
                if len(faces_data) < 150:
                    faces_data.append(resized_img)

    # Draw boxes around detected faces
    for (x, y, w, h) in face_rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for detected faces

    # Calculate remaining time
    elapsed_time = time.time() - start_time
    remaining_time = max_duration - elapsed_time
    if remaining_time < 0:
        remaining_time = 0

    # Create a frame for text display
    text_frame = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(text_frame, "Face Detection in Progress...", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 215, 0), 3)
    cv2.putText(text_frame, f"Time left: {int(remaining_time)} seconds", (20, 110), cv2.FONT_HERSHEY_COMPLEX, 1.5, (135, 206, 250), 3)

    # Blend the text frame with the main frame
    frame[:200, :] = text_frame

    # Display the final output
    cv2.imshow("Face Detection", frame)

    # Check for timeout or 'q' to quit
    if len(faces_data) >= 150 or remaining_time <= 0:
        print(f"Data collection complete for {name}.")
        break

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Prepare faces data for saving
faces_data = np.asarray(faces_data)

# Reshape the face data to consistent dimensions (50x50 pixels, 3 color channels)
faces_data = faces_data.reshape(len(faces_data), -1)

# Path to store the name and face data
data_folder = '/Users/aayushpande/MULTIPLE_FACES/'

# Saving name and faces data
names_file = os.path.join(data_folder, 'names.pkl')
faces_file = os.path.join(data_folder, 'faces_data.pkl')

# Save names
if not os.path.exists(names_file) or os.path.getsize(names_file) == 0:
    names = [name] * len(faces_data)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * len(faces_data))
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save face data, ensuring same dimensions
if not os.path.exists(faces_file) or os.path.getsize(faces_file) == 0:
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    
    # Ensure the data being concatenated has the same dimensions
    if faces.shape[1] == faces_data.shape[1]:
        faces = np.concatenate((faces, faces_data), axis=0)
    else:
        print("Mismatch in face data dimensions. Cannot concatenate.")
    
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

print("Faces and names saved successfully!")
