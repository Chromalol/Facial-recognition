import numpy as np
import os, cv2
from PIL import Image

def train_classifier(name):
    """
    Trains a custom classifier to recognize a face based on the specified name.
    :param name: the name of the person whose face the classifier will recognize
    """
    # Get the path to the directory containing the images for the specified name
    path = os.path.join(os.getcwd()+"/data/"+name+"/")
    images = []
    labels = []
    ids = []
    picture_filenames = {}

    # Iterate through the images in the directory and store them in a numpy array along with their corresponding labels (ids)
    for root, dirs, files in os.walk(path):
            picture_filenames = files
    for pic in picture_filenames:
        imgpath = path+pic
        img = Image.open(imgpath).convert('L')
        image_np = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        images.append(image_np)
        ids.append(id)

    ids = np.array(ids)

    # Train the classifier and save it to a file
    classifier = cv2.face.LBPHFaceRecognizer_create()
    classifier.train(images, ids)
    classifier.write("./data/classifiers/"+name+"_classifier.xml")
