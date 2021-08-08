import matplotlib.pyplot as plt
import torch
from src.features.detectors import *
import numpy as np
from glob import glob
import os
import timeit

start = timeit.default_timer()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# load filenames for human and dog images
human_files = np.array(glob("data/lfw/*/*"))
dog_files = np.array(glob("data/dog_images/*/*/*"))

model = torch.load('src/models/model_.pt')
face_cascade = cv2.CascadeClassifier('cv2/haarcascade_frontalface_alt.xml')

def run_app(img_path):
    # import and show image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()

    x_transf = image_transforms(img_path)

    ## handle cases for a human face, dog, and neither
    dog = dog_detector_inc(x_transf)
    human = face_detector(img_path, face_cascade)
    breed = predict_breed_transfer(x_transf, model)

    if dog == True:
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return print('dog breed detected:', breed)

    elif human == True:
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return print('dog breed resembling human face:', breed)

    else:
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return print('error: No dog nor human face detected')


run_app(dog_files[0])

