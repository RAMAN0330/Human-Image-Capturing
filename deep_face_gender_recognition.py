import os
import cv2
from deepface import DeepFace

# Directory containing images
image_directory = '/home/satoshi/Desktop/Images'

# Initialize an empty dictionary to store gender predictions
gender_predictions = {}

# Loop through each image in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Load the image
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        # Perform face detection using OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]

            # Use DeepFace for gender prediction
            result = DeepFace.analyze(face, enforce_detection=False, actions=['gender'])

            # Get the gender prediction
            gender = result[0]['gender']

            # Find the key with the maximum value
            max_gender = max(gender, key=lambda k: gender[k])

            # Print the key
            print(f'The gender with the maximum count is: {max_gender}')

            # Update the gender predictions dictionary
            gender_predictions[image_path] = max_gender

# Print the gender predictions
print('Gender predictions:')
for image_path, gender in gender_predictions.items():
    print(f'{image_path}: {gender}')