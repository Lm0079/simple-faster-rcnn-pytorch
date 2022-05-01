import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import cv2
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
data = '/user/work/gh18931/diss/datasets/zebra_test_set/'
model_names = ['/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints_epoch_14/fasterrcnn_04171116_0.8256965596830876']
for model_name in model_names :
	print("model loading")
	trainer.load(model_name)

	model_n = model_name.split("/")[-1]
	output_p = f"cropped/{model_n}"
	isExist = os.path.exists(output_p)
	if not isExist:
		os.makedirs(output_p)
	valid_img = ".jpg"
	for f in os.listdir(data):
		ext = os.path.splitext(f)[1]
		if ext.lower() != valid_img:
			continue
		print(f)
		file = data + f
		img = read_image(file)
		img = t.from_numpy(img)[None]

		opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
		_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
		
		for labels in _labels:
			
			found = False
			for count ,label in enumerate(labels):
				image = cv2.imread(file)

				if label == 1 and not found:
					bbox = _bboxes[0][count]
					
					found=True
					cropped_img = image[int(bbox[0]) : int(bbox[2]-bbox[0]), int(bbox[1]):int(bbox[3] - bbox[1])]
					output_f = os.path.join(output_p , f"{f}")
					if not cropped_img.any() : 
						pass
					else:
						#print(cropped_img)
						cv2.imwrite(output_f,cropped_img)

					
					
	