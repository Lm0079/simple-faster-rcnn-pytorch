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
data = '/user/work/gh18931/diss/test/imgs/'
model_names = ["/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_05011331_0.8941059978840188","/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_05011328_0.8547641971555016","/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_05011327_0.8524188234686645","/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_05011321_0.8895999193454125","/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_05011319_0.8258197446281452"]
for model_name in model_names :
	print("model loading")
	trainer.load(model_name)

	model_n = model_name.split("/")[-1]
	output_p = f"output_3/{model_n}/"
	isExist = os.path.exists(output_p)
	if not isExist:
		os.makedirs(output_p)
	valid_img = ".jpg"
	for f in os.listdir(data):
		ext = os.path.splitext(f)[1]
		if ext.lower() != valid_img:
			continue
		#print(f)
		file = data + f
		img = read_image(file)
		img = t.from_numpy(img)[None]

		opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
		_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)

		for labels in _labels:
			found = False
			for count ,label in enumerate(labels):
				if label == 1 and not found:
					image = cv2.imread(file)
					print(file)
					ax = vis_bbox(at.tonumpy(img[0]),
						[list(_bboxes[0][count])],
						[_labels[0][count]],
						[_scores[0][count]])
					ax.figure.savefig(f"{output_p}{f}")
					bbox = _bboxes[0][count]
					found=True
					print(bbox)
					cropped_img = image[int(bbox[0]) : int(bbox[2]), int(bbox[1]):int(bbox[3] )]
					cv2.imwrite(f"{output_p}cropped_{f}",cropped_img)
