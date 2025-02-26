import cv2

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Ask the user to input a name first
face_name = input("Enter your name: ")

# Dictionary to store faces (this example just uses a single name, but you can extend it for more)
face_names = {}

# Store the name in the dictionary (initially with a None value, to be updated later)
face_names[face_name] = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale for better face detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the name on the frame near the detected face
        cv2.putText(frame, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit the application
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
