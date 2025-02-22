#!/usr/bin/env python3
# flake8: noqa: E501

import flask
import torch
from torchvision import transforms
from PIL import Image
from time import time

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
    t0 = time()
    data = flask.request.files['image']
    img = Image.open(data.stream)
    t1 = time()
    with torch.no_grad():
        myimg = loader(img).float()
        result = model(myimg[None, :, :, :])[0][:5]
        probs = torch.nn.functional.softmax(result, dim=0)
    t2 = time()
    outstr = [f'{c:<10}: {probs[i]:.3f}' for i, c in enumerate(classes)]
    return '\n'.join(outstr)+'\n'+f'Times: {t1-t0:.3f} loadig, {t2-t1:.3f} processing.\n'


@server.route('/species', methods=['POST'])
def identify_species():
    classes = ['Atl sal', 'Pink sal', 'Rainbow', 'Arctic Char', 'Trout']
    return run_model(species_model, classes)


@server.route('/type', methods=['POST'])
def identify_type():
    classes = ['Oppdrett', 'Vill', 'O. Ørret']
    return run_model(type_model, classes)
