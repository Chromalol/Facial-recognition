import cv2
import os

def capture_faces(label):
    # Create directory for images if it doesn't already exist
    path = "./data/" + label
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')

    # Initialize variables and face detector
    num_captured = 0
    face_detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    # Loop until user exits or maximum number of images is reached
    while True:
        # Capture frame from video feed
        ret, frame = video_capture.read()

        # Convert to grayscale and detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around detected faces and display image
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(frame, f"{num_captured} images captured", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            face_img = frame[y:y+h, x:x+w]
        cv2.imshow("Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # Save detected face to file
        try:
            cv2.imwrite(f"{path}/{num_captured}{label}.jpg", face_img)
            num_captured += 1
        except:
            pass

        # Break loop if user presses 'q' or maximum number of images reached
        if key == ord("q") or key == 27 or num_captured > 310:
            break

    # Clean up resources
    video_capture.release()
    cv2.destroyAllWindows()
    return num_captured
