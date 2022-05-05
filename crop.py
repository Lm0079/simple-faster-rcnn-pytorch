#!/usr/bin/env python3
import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
import argparse
from pathlib import Path
from utils import array_tool as at
import cv2
parser = argparse.ArgumentParser(
	description=" Ablation Cropper",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
	"--data",
	type=Path,
	help=""
)
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
def main(args):
	model_names = ['/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_05011331_0.8941059978840188']
	for model_name in model_names :
		print("model loading")
		trainer.load(model_name)

		valid_img = ".jpg"
		for f in os.listdir(args.data):
			ext = os.path.splitext(f)[1]
			if ext.lower() != valid_img:
				continue
			if "cropped" in f:
				continue
			print(f)
			file = str(args.data) +"/"+ f
			img = read_image(file)
			img = t.from_numpy(img)[None]

			opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
			_bboxes, _labels, _= trainer.faster_rcnn.predict(img,visualize=True)
			
			for labels in _labels:
				
				found = False
				for count ,label in enumerate(labels):
					image = cv2.imread(file)

					if label == 1 and not found:
						bbox = _bboxes[0][count]
						
						found=True
						cropped_img = image[int(bbox[0]) : int(bbox[2]), int(bbox[1]):int(bbox[3] )]
						output_f = os.path.join(str(args.data) , f"cropped_{f}")
						if not cropped_img.any() : 
							pass
						else:
							cv2.imwrite(output_f,cropped_img)

					
if __name__ == "__main__":
	main(parser.parse_args())