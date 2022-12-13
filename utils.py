import torch, torchvision
import numpy as np

def evaluate(model, data_loader, device, mAP_list=None):
    mAP = 0
    return mAP

def draw_box(img_path, box):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    im = Image.open(img_path)
    w, h = im.size

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    xmin = (box[0] - box[2]/2)*w
    ymin = (box[1] - box[3]/2)*h
    width = box[2]*w
    height = box[3]*h
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()



