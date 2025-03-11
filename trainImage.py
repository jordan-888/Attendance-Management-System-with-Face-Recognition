import os
import cv2
import numpy as np
from PIL import Image

def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message, text_to_speech):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, Id = getImagesAndLables(trainimage_path)
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    text_to_speech("Image Trained")
    message.configure(text="Image Trained")

def getImagesAndLables(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        if os.path.isfile(imagePath):  # Ensure it's a file
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split("_")[1])
            faces.append(imageNp)
            Ids.append(Id)
    return faces, Ids