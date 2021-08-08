from PIL import Image
from torch.autograd import Variable
from torchvision import transforms,models
import torch.nn.functional as F
import pandas as pd
import cv2


def image_transforms(img_path):
    image = Image.open(img_path)
    trans0 = transforms.Resize(224)
    trans = transforms.CenterCrop(224)
    trans1 = transforms.ToTensor()
    trans2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x_transf = trans2(trans1(trans(trans0(image)))).unsqueeze(0)

    return x_transf


def VGG16_predict(x_transf):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    model = models.vgg16(pretrained=True)
    model.eval()
    output = model(x_transf)
    probs = F.softmax(output, dim=1)

    return probs.argmax()  # predicted class index


def dog_detector(img_path):
    '''
    Dog detector

    Args:
        img_path: path to an image

    Returns:
        True if image is a dog
    '''
    output = VGG16_predict(img_path)
    if output in range(151,268):
        return True
    else:
        return False


def predict_breed_transfer(x_transf, model):
    class_names = list(pd.read_csv('class_names.csv')['0'])

    # select model to use
    prob = model.forward(Variable(x_transf.cuda())).cpu()

    return class_names[prob.data.numpy().argmax()]


def face_detector(img_path, model):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray)
    print('faces',faces)
    return len(faces) > 0


def inception_predict(x_transf):

    model = models.inception_v3(pretrained=True)
    model.eval()
    output = model(x_transf)
    probs = F.softmax(output, dim=1)
    class_index = probs.argmax()

    return class_index  # predicted class index


def dog_detector_inc(x_transf):
    '''
    Dog detector_inc

    Args:
        img_path: path to an image

    Returns:
        True if image is a dog'''
    class_index = inception_predict(x_transf)

    if class_index.item() in range(151, 268):
        return True
    else:
        return False