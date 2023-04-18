import cv2 # for capturing and displaying video
import numpy as np # for working with arrays
from mtcnn.mtcnn import MTCNN # for detecting faces
from keras.models import load_model # for loading a trained model
from keras.preprocessing.image import img_to_array # for preprocessing images

# Load the video capturer
capturer = cv2.VideoCapture(0)

# Create an MTCNN face detector
detector = MTCNN()

# Load the trained emotion model
emotion_model_path = "./data/_mini_XCEPTION.106-0.65.hdf5"
emotion_classifier = load_model(emotion_model_path, compile=False)

# Load the trained age and gender models
age_proto = "./data/age_deploy.prototxt"
age_model = "./data/age_net.caffemodel"
gender_proto = "./data/gender_deploy.prototxt"
gender_model = "./data/gender_net.caffemodel"
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Constants for age and gender prediction
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Constants for emotion prediction
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

def age_and_gender():
    """
    Predicts the age and gender of all detected faces in the video frame.
    """
    while True:
        # Read a frame from the video capturer
        ret, frame = capturer.read()

        # Convert the frame to RGB
        default_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(image=default_frame, scaleFactor=1.3, minNeighbors=5)

        # For each detected face
        for (x, y, w, h) in faces:



            def show_emotion():
# Open the default camera
                camera = cv2.VideoCapture(0)

# Load MTCNN face detector
face_detector = MTCNN()

# Load the emotion model
emotion_model_path = "./data/_mini_XCEPTION.106-0.65.hdf5"
emotion_classifier = load_model(emotion_model_path, compile=False)

# Emotions that the model can recognize
emotions = ["angry","disgust","scared","happy","sad","surprised","neutral"]

# Haar cascade classifier for detecting faces
face_classifier = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

while True:
    # Capture frame from camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Get the ROI (region of interest) of the face
        roi = gray[y:y+h, x:x+w]
        # Resize the ROI to (48, 48)
        roi = cv2.resize(roi, (48, 48))
        # Normalize the ROI
        roi = roi.astype("float") / 255.0
        # Convert the ROI to a 4D tensor
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Make a prediction on the ROI
        preds = emotion_classifier.predict(roi)[0]
        # Get the label with the highest probability
        label = emotions[preds.argmax()]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Add a label with the emotion text above the rectangle
        cv2.putText(frame, f"{label}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

    # Show the frame with the detected faces and emotions
    cv2.imshow("Emotion Detection", frame)
    # Wait for the user to press a key
    key = cv2.waitKey(1) & 0xFF
    # If the user pressed 'q', break the loop
    if key == ord("q"):
            break

# Destroy all windows and close the camera
cv2.destroyAllWindows()
camera.release()
