import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from iresnet import iresnet100
import numpy as np


def main():

	device='cuda' if torch.cuda.is_available() else 'cpu'
	
	arcface = iresnet100(pretrained=False, fp16=False)
	arcface.load_state_dict(torch.load('checkpoints/arcface.pt', map_location='cpu'))
	arcface=arcface.to(device)
	arcface.eval()


	blendface = iresnet100(pretrained=False, fp16=False)
	blendface.load_state_dict(torch.load('checkpoints/blendface.pt', map_location='cpu'))
	blendface=blendface.to(device)
	blendface.eval()
	
	
	
	with torch.no_grad():
	
		img_anc=np.array(Image.open('images/anchor.png'))
		img_pos=np.array(Image.open('images/positive.png'))
		img_neg=np.array(Image.open('images/negative.png'))
		img_swp=np.array(Image.open('images/swapped.png'))

		img_anc=torch.tensor(img_anc).to(device).permute(2,0,1).unsqueeze(0)/255
		img_pos=torch.tensor(img_pos).to(device).permute(2,0,1).unsqueeze(0)/255
		img_neg=torch.tensor(img_neg).to(device).permute(2,0,1).unsqueeze(0)/255
		img_swp=torch.tensor(img_swp).to(device).permute(2,0,1).unsqueeze(0)/255

		img_anc=(img_anc-0.5)/0.5
		img_pos=(img_pos-0.5)/0.5
		img_neg=(img_neg-0.5)/0.5
		img_swp=(img_swp-0.5)/0.5
		
		#arcface
		vec_anc=F.normalize(arcface(img_anc))
		vec_pos=F.normalize(arcface(img_pos))
		vec_neg=F.normalize(arcface(img_neg))
		vec_swp=F.normalize(arcface(img_swp))

		sim_pos=nn.CosineSimilarity()(vec_anc,vec_pos).item()
		sim_neg=nn.CosineSimilarity()(vec_anc,vec_neg).item()
		sim_swp=nn.CosineSimilarity()(vec_anc,vec_swp).item()
	
		print(f'ArcFace| Positive: {sim_pos:0.4f}, Negative: {sim_neg:0.4f}, Swapped: {sim_swp:0.4f}')


		#blendface
		vec_anc=F.normalize(blendface(img_anc))
		vec_pos=F.normalize(blendface(img_pos))
		vec_neg=F.normalize(blendface(img_neg))
		vec_swp=F.normalize(blendface(img_swp))

		sim_pos=nn.CosineSimilarity()(vec_anc,vec_pos).item()
		sim_neg=nn.CosineSimilarity()(vec_anc,vec_neg).item()
		sim_swp=nn.CosineSimilarity()(vec_anc,vec_swp).item()
	
		print(f'BlendFace| Positive: {sim_pos:0.4f}, Negative: {sim_neg:0.4f}, Swapped: {sim_swp:0.4f}')




			
if __name__=='__main__':
	main()
