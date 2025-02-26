import cv2

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Ask for the name when the program starts
name = input("Please enter your name: ")

# Dictionary to store names associated with faces
face_names = {name: None}

print(f"Hello {name}! Now let's start face detection...")

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
        
        # Display the name near the face
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Show the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit the application
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
