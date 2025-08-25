# 🧠 Real-Time Face, Eye, and Smile Detection using OpenCV with Enhanced Features

import cv2
import time
import datetime
import os

# === Initialization Section ===

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Check for successful loading
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("[ERROR] Haar cascades could not be loaded.")
    exit()

# Create output directory for snapshots
output_dir = "snapshots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

# Variables for FPS calculation
prev_frame_time = time.time()
snapshot_counter = 0

print("[INFO] Press 'q' to quit, 's' to take a snapshot.")

# === Main Loop for Real-Time Detection ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )

    # Draw detections
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of interest for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

        # Detect smile within the face
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)

    # === Calculate and display FPS ===
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    # === Overlay text on the frame ===
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
    cv2.putText(frame, "Press 'q' to quit | 's' to snapshot",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # === Display the frame ===
    cv2.imshow("Enhanced Face Detection", frame)

    # === Handle Key Events ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quitting...")
        break
    elif key == ord('s'):
        # Save snapshot with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"{output_dir}/snapshot_{timestamp}_{snapshot_counter}.png"
        cv2.imwrite(snapshot_name, frame)
        snapshot_counter += 1
        print(f"[INFO] Snapshot saved: {snapshot_name}")

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program ended.")

