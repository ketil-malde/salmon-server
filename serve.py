#!/usr/bin/env python3

import flask
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

server = flask.Flask('salmon-server')

species_model = torch.load('species.pth', weights_only=False, map_location=torch.device('cpu'))
species_model.eval()

type_model = torch.load('type.pth', weights_only=False, map_location=torch.device('cpu'))
type_model.eval()

loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def run_model(model, classes):
    data = flask.request.files['image']
    img = Image.open(data.stream)
    with torch.no_grad():
        myimg = loader(img).float()
        result = model(myimg[None,:,:,:])[0][:5]
        probs = torch.nn.functional.softmax(result, dim=0)

    outstr = [f'{c}: {probs[i]:.3f}' for i,c in enumerate(classes)]
    return(f'Result:\n'+'\n'.join(outstr)+'\n')

@server.route('/species', methods=['POST'])
def identify_species():
    classes = ['Atl sal', 'Pink sal', 'Rainbow', 'Arctic Char', 'Trout']
    return run_model(species_model, classes)

@server.route('/type', methods=['POST'])
def identify_type():
    classes = ['Oppdrett', 'Vill', 'O. Ørret']
    return run_model(type_model, classes)
