import cv2
import os
import numpy as np
import serial
from datetime import datetime
import time  # Import time for adding delay
import imgaug.augmenters as iaa  # Import imgaug for data augmentation



# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare lists to hold training data
face_samples = []
ids = []

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
    iaa.Affine(rotate=(-30, 30)),  # Rotate by -30 to +30 degrees
    iaa.Multiply((0.8, 1.2)),  # Change brightness
    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add Gaussian noise
    iaa.Crop(percent=(0, 0.1)),  # Randomly crop images
])

# Load training images from the 'training_images' directory
def load_training_images(training_dir):
    for filename in os.listdir(training_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            img_path = os.path.join(training_dir, filename)
            image = cv2.imread(img_path)

            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect face in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            # For each detected face, append the face image and ID
            for (x, y, w, h) in faces:
                face = gray_image[y:y + h, x:x + w]
                face_samples.append(face)
                ids.append(int(filename.split('_')[0]))  # Use the file name (ID)

                # Apply augmentation
                augmented_faces = seq(images=[face] * 5)  # Create 5 augmented versions of the face

                for augmented_face in augmented_faces:
                    face_samples.append(augmented_face)
                    ids.append(int(filename.split('_')[0]))  # Use the same ID for augmented images

    return ids

# Load training images
training_dir = r"C:\Users\SANSKRITI VERMA\PycharmProjects\pythonpractice\pythonProject\training_images"
ids = load_training_images(training_dir)

# Train the recognizer with the collected face samples
recognizer.train(face_samples, np.array(ids))

# Save the trained model
recognizer.save('face_recognizer.yml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to mark attendance (detected face)
def mark_attendance(name):
    with open('attendance.csv', 'a') as file:
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        file.write(f'{name},{time_str}\n')

# Dictionary to map IDs to names
id_to_name = {0: "sanskriti", 1: "lucky", 2: "anushka", 3: "jahnavi"}  # Updated ID to name mapping

# Set to keep track of already recorded attendance
recorded_attendance = set()

# Capture duration per face (in seconds)
display_duration = 5  # Time to display each face before checking for attendance again
recognition_display_time = 3  # Time to display the name after recognizing
last_recognized_time = time.time()  # Track the time since the last recognition

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    recognized_name = None  # Variable to hold the name of the recognized person

    for (x, y, w, h) in faces:
        # Recognize the face
        id_, confidence = recognizer.predict(gray_frame[y:y + h, x:x + w])

        if confidence < 70:  # Lower confidence means better recognition
            name = id_to_name.get(id_, "Unknown")

            # Check if recognized name is not "Unknown" and attendance not recorded in this session
            if name != "Unknown" and name not in recorded_attendance:
                current_time = time.time()
                # Mark attendance once within the specified duration
                if current_time - last_recognized_time > display_duration:
                    mark_attendance(name)
                    recorded_attendance.add(name)  # Mark this name as recorded
                    print(f'Attendance recorded for {name}')
                    last_recognized_time = current_time
                    recognized_name = name  # Store the recognized name for display

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            time.sleep(3)


        else:
            name = "Unknown"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)


    # If a name was recognized, show it for a limited time
    if recognized_name:
        # Display the name for a certain period
        for t in range(recognition_display_time):
            cv2.imshow('Face Recognition Attendance System', frame)
            cv2.putText(frame, recognized_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.waitKey(1000)  # Wait for 1 second
            # This will exit the loop and proceed to release the camera

        break  # Exit after displaying the recognized name for a limited time

    # Display the frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Delay to slow down frame capture
    time.sleep(0.1)

    # Break the loop if 'q' is pressedT
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
