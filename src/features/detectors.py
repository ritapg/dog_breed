from PIL import Image
from torch.autograd import Variable
from torchvision import transforms,models
import torch.nn.functional as F
import pandas as pd


def image_transforms(img_path):
    '''
    Use transformation of the images to be as per models input

    Args:
        img_path: path to an image

    Returns:
        tensor of the image
    '''
    image = Image.open(img_path)
    trans0 = transforms.Resize(224)
    trans = transforms.CenterCrop(224)
    trans1 = transforms.ToTensor()
    trans2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x_transf = trans2(trans1(trans(trans0(image)))).unsqueeze(0)

    return x_transf


def inception_predict(x_transf):

    model = models.inception_v3(pretrained=True)
    model.eval()
    output = model(x_transf)
    probs = F.softmax(output, dim=1)
    class_index = probs.argmax()

    return class_index  # predicted class index


def dog_detector_inc(x_transf):
    '''
    Dog detector using pre-trained inception model

    Args:
        img_path: path to an image

    Returns:
        True if image is a dog'''
    class_index = inception_predict(x_transf)

    if class_index.item() in range(151, 268):
        return True
    else:
        return False


def predict_breed_transfer(x_transf, model):
    '''
    Dog breed classifer

    Args:
        img_path: path to an image

    Returns:
        breed of the dog'''

    class_names = list(pd.read_csv('class_names.csv')['0'])

    # select model to use
    prob = model.forward(Variable(x_transf.cuda())).cpu()

    return class_names[prob.data.numpy().argmax()]
