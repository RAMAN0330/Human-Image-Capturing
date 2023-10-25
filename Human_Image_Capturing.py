import cv2
import os

# Set up the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create the save directory if it doesn't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

# Start capturing the video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-50, y-50), (x+w+50, y+h+50), (255, 0, 0), 2)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Check if the button is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
           # Save the captured face
          for (x, y, w, h) in faces:
              face = frame[y:y+h+50, x:x+w+50]
              cv2.imwrite(f'faces/face_{len(os.listdir("faces"))+1}.jpg', face)
    

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()