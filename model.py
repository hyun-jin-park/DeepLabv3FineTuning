""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(outputchannels=1, pretrain=True):
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrain, progress=pretrain)
    # Added a Tanh activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    return model
