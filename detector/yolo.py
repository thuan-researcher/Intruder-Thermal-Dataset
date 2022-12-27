import torch
import os
import shutil
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import myUtils

IMAGESDIR = "./img"


def loadModel(imageDir,cropImagesDir):
    #load model
    model = torch.hub.load("ultralytics/yolov5", 'custom', path="")
    print("Successfully load model")

    # input image to model

    for imagePath in imagesPathList:
        img = Image.open(imagePath)
        result = model(img)

# ham ve boundingbox

def main():
    loadModel(IMAGESDIR)


if __name__ == "__main__":
    main()