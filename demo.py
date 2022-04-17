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
file = '/user/work/gh18931/diss/datasets/zebra_test_set/804454c0-4f2f-407a-a4fa-d2b38a88d50c_female_96c17794-e32d-4aa4-943a-193e17426ce5_right.jpg'
trainer.load('/user/work/gh18931/diss/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_04162145_0.7899986061855276')
img = read_image(file)
img = t.from_numpy(img)[None]

opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
ax = vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
ax.figure.savefig("hi.png")
print("hi")
# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it