import argparse
from PIL import Image

import torch
import torchvision
from torchvision import transforms

from blendswap import BlendSwap
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("-w","--weight_path", type=str, required=True)
parser.add_argument("-t","--target_image", type=str, default='examples/target.png')
parser.add_argument("-s","--source_image", type=str, default='examples/source.png')
parser.add_argument("-o","--output_path", type=str, default="examples/output.png")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = BlendSwap()
model.load_state_dict(torch.load(args.weight_path,map_location='cpu'))
model.eval()
model.to(device)

target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(target_img, source_img)
Image.fromarray((output.permute(0,2,3,1)[0].cpu().data.numpy()*255).astype(np.uint8)).save(args.output_path)