import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
data = '/user/work/gh18931/diss/datasets/zebra_test_set/'
model_names = ['/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04171114_0.7569560460681852','/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04171115_0.7898412502829089','/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04171116_0.8256965596830876','/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04171116_0.8263813451381766','/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04171119_0.8714583358748896','/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04171121_0.8831425095611765']
for model_name in model_names :
	print("model loading")
	trainer.load(model_name)

	model_n = model_name.split("/")[-1]
	output_p = f"output/{model_n}/"
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
		ax = vis_bbox(at.tonumpy(img[0]),
				at.tonumpy(_bboxes[0]),
				at.tonumpy(_labels[0]).reshape(-1),
				at.tonumpy(_scores[0]).reshape(-1))
		ax.figure.savefig(f"{output_p}{f}")
		

	
	# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it