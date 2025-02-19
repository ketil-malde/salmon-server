#!/usr/bin/env python3

import flask
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

server = flask.Flask('salmon-server')
model = torch.load('model.pth', weights_only=False, map_location=torch.device('cpu'))
model.eval()

classes = ['Atl sal ', 'Pink sal', 'Rb trout', 'Char    ', 'Trout   ']

loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@server.route('/', methods=['POST'])
def identify():
    data = flask.request.files['image']
    img = Image.open(data.stream)
    with torch.no_grad():
        myimg = loader(img).float()
        result = model(myimg[None,:,:,:])[0][:5]
        probs = torch.nn.functional.softmax(result, dim=0)

    outstr = [f'{c}: {probs[i]:.3f}' for i,c in enumerate(classes)]
    return(f'Result:\n'+'\n'.join(outstr)+'\n')
